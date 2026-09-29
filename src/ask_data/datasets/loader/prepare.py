"""DataFrames -> typed, validated tables, entirely in memory.

Everything that can be checked without Postgres is checked here, and *all* problems are
reported together, so a bad dataset fails before the load transaction even starts:
unknown yaml columns, values that don't fit a declared type, duplicate/NULL primary keys,
foreign keys that point at missing rows or at columns of an incompatible type.
"""

from dataclasses import dataclass, field

import pandas as pd

from ask_data.datasets.config import DatasetCfg, TableCfg
from ask_data.datasets.loader.infer import (
    Column,
    ConversionError,
    ParseOptions,
    convert,
    infer,
    to_strings,
    type_family,
)
from ask_data.datasets.loader.sanitize import sanitize_columns
from ask_data.datasets.loader.sources import null_values

MAX_EXAMPLES = 5


class LoadError(ValueError):
    """One or more problems with the dataset; the message lists all of them."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(f"  - {p}" for p in problems))


@dataclass
class PreparedTable:
    name: str
    cfg: TableCfg
    columns: list[Column]
    renamed: dict[str, str] = field(default_factory=dict)  # source header -> column name

    @property
    def row_count(self) -> int:
        return len(self.columns[0].values) if self.columns else 0

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def rows(self, names: list[str]) -> list[tuple]:
        return list(zip(*(self.column(n).values for n in names), strict=True))


def prepare_tables(cfg: DatasetCfg, frames: dict[str, pd.DataFrame]) -> dict[str, PreparedTable]:
    opts = ParseOptions(
        thousands=cfg.source.options.get("thousands"),
        dayfirst=bool(cfg.source.options.get("date_dayfirst", False)),
    )
    nulls = null_values(cfg)
    problems: list[str] = []
    tables: dict[str, PreparedTable] = {}

    for name, table_cfg in cfg.tables.items():
        table = _prepare_table(name, table_cfg, frames[name], opts, nulls, cfg, problems)
        if table is not None:
            tables[name] = table

    if not problems:  # cross-table checks need every table typed
        for name, table in tables.items():
            problems += _primary_key_problems(table)
            problems += _foreign_key_problems(table, tables)
    if problems:
        raise LoadError(problems)
    return tables


def _prepare_table(
    name: str,
    table_cfg: TableCfg,
    df: pd.DataFrame,
    opts: ParseOptions,
    nulls: frozenset[str],
    cfg: DatasetCfg,
    problems: list[str],
) -> PreparedTable | None:
    headers = list(df.columns)
    clean = sanitize_columns(headers)
    if not clean:
        problems.append(f"{name}: source has no columns")
        return None
    renamed = {str(h): c for h, c in zip(headers, clean, strict=True) if str(h) != c}

    referenced = {
        **{c: "columns:" for c in table_cfg.columns},
        **{c: "primary_key" for c in table_cfg.primary_key},
        **{c: "foreign_keys" for fk in table_cfg.foreign_keys for c in fk.columns},
        **{c: "indexes" for t, cols in cfg.parsed_indexes if t == name for c in cols},
    }
    missing = [f"{c!r} ({where})" for c, where in referenced.items() if c not in clean]
    if missing:
        problems.append(
            f"{name}: dataset.yaml names column(s) not in the data: {', '.join(missing)}; "
            f"available: {', '.join(clean)}"
        )
        return None

    columns: list[Column] = []
    for header, col_name in zip(headers, clean, strict=True):
        values = to_strings(df[header], nulls)
        declared = table_cfg.columns.get(col_name)
        try:
            if declared and declared.type:
                columns.append(convert(col_name, values, declared.type, opts))
            else:
                columns.append(infer(col_name, values, opts, cfg.load.infer_threshold))
        except ConversionError as e:
            problems.append(f"{name}: {e}")
    return PreparedTable(name=name, cfg=table_cfg, columns=columns, renamed=renamed)


def _primary_key_problems(table: PreparedTable) -> list[str]:
    pk = table.cfg.primary_key
    if not pk:
        return []
    keys = table.rows(pk)
    problems = []
    null_rows = sum(any(v is None for v in k) for k in keys)
    if null_rows:
        problems.append(f"{table.name}: {null_rows} row(s) have NULL in primary key {pk}")
    seen, dupes = set(), []
    for k in keys:
        if k in seen:
            dupes.append(k)
        seen.add(k)
    if dupes:
        examples = ", ".join(_fmt(k) for k in list(dict.fromkeys(dupes))[:MAX_EXAMPLES])
        problems.append(
            f"{table.name}: {len(dupes)} duplicate primary key value(s) in {pk}, e.g. {examples}"
        )
    return problems


def _foreign_key_problems(table: PreparedTable, tables: dict[str, PreparedTable]) -> list[str]:
    problems = []
    for fk in table.cfg.foreign_keys:
        target = tables[fk.ref_table]
        label = f"{table.name}({', '.join(fk.columns)}) -> {fk.references}"
        missing = [c for c in fk.ref_columns if c not in target.column_names()]
        if missing:
            problems.append(f"{label}: referenced column(s) {missing} not in {target.name}")
            continue
        if set(fk.ref_columns) != set(target.cfg.primary_key):
            problems.append(
                f"{label}: a foreign key must reference the primary key of {target.name} "
                f"({target.cfg.primary_key or 'none declared'})"
            )
            continue
        mismatched = [
            f"{c} {table.column(c).pg_type} vs {r} {target.column(r).pg_type}"
            for c, r in zip(fk.columns, fk.ref_columns, strict=True)
            if type_family(table.column(c).pg_type) != type_family(target.column(r).pg_type)
        ]
        if mismatched:
            problems.append(
                f"{label}: column types differ ({'; '.join(mismatched)}); "
                f"set the same `type:` on both sides in dataset.yaml"
            )
            continue
        parent = set(target.rows(fk.ref_columns))
        orphans = [k for k in table.rows(fk.columns) if None not in k and k not in parent]
        if orphans:
            examples = ", ".join(_fmt(k) for k in list(dict.fromkeys(orphans))[:MAX_EXAMPLES])
            problems.append(
                f"{label}: {len(orphans)} row(s) reference a missing {target.name} row, "
                f"e.g. {examples}; fix the data or remove the foreign key from dataset.yaml"
            )
    return problems


def _fmt(key: tuple) -> str:
    return repr(key[0]) if len(key) == 1 else repr(key)
