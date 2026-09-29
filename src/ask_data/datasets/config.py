"""DatasetCfg: the validated shape of a datasets/<name>/dataset.yaml file.

Identifiers (dataset name, schema, table names, column names) are validated here so
that everywhere they're later interpolated into SQL, they go through psycopg.sql.Identifier
on values already known to be safe — never an f-string on raw yaml text.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

IDENT = r"^[a-z][a-z0-9_]{0,30}$"
IDENT_RE = re.compile(IDENT)

RESERVED_SCHEMAS = {"app", "public", "pg_catalog", "information_schema"}

BASE_TYPES = {
    "text",
    "int",
    "bigint",
    "numeric",
    "double precision",
    "boolean",
    "date",
    "timestamp",
    "timestamptz",
}
NUMERIC_WITH_PRECISION_RE = re.compile(r"^numeric\(\s*\d{1,3}\s*,\s*\d{1,3}\s*\)$")


def is_allowed_type(value: str) -> bool:
    return value in BASE_TYPES or bool(NUMERIC_WITH_PRECISION_RE.match(value))


class ColumnCfg(BaseModel):
    type: str | None = None
    comment: str | None = None

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str | None) -> str | None:
        if v is not None and not is_allowed_type(v):
            raise ValueError(f"unsupported column type: {v!r}")
        return v


def parse_table_ref(ref: str) -> tuple[str, list[str]]:
    """Split "table(col, ...)" into ("table", ["col", ...]) — used for FK references and indexes."""
    if "(" not in ref or not ref.endswith(")"):
        raise ValueError(f"must look like 'table(col, ...)': {ref!r}")
    table, inner = ref.split("(", 1)
    return table.strip(), [c.strip() for c in inner[:-1].split(",")]


class ForeignKey(BaseModel):
    columns: list[str]
    references: str  # "table(col, ...)"

    @property
    def ref_table(self) -> str:
        return parse_table_ref(self.references)[0]

    @property
    def ref_columns(self) -> list[str]:
        return parse_table_ref(self.references)[1]

    @model_validator(mode="after")
    def _checks(self) -> "ForeignKey":
        if not self.columns:
            raise ValueError("foreign_keys entry needs at least one column")
        for c in self.columns:
            if not IDENT_RE.match(c):
                raise ValueError(f"invalid column identifier in foreign key: {c!r}")
        if "(" not in self.references or not self.references.endswith(")"):
            raise ValueError(f"references must look like 'table(col, ...)': {self.references!r}")
        if not IDENT_RE.match(self.ref_table):
            raise ValueError(f"invalid referenced table identifier: {self.ref_table!r}")
        ref_cols = self.ref_columns
        if len(ref_cols) != len(self.columns):
            raise ValueError(
                f"foreign key has {len(self.columns)} column(s) but references "
                f"{len(ref_cols)} column(s): {self.references!r}"
            )
        for c in ref_cols:
            if not IDENT_RE.match(c):
                raise ValueError(f"invalid referenced column identifier: {c!r}")
        return self


class TableCfg(BaseModel):
    file: str | None = None
    comment: str | None = None
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    columns: dict[str, ColumnCfg] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _checks(self) -> "TableCfg":
        for c in self.primary_key:
            if not IDENT_RE.match(c):
                raise ValueError(f"invalid primary key column identifier: {c!r}")
        for c in self.columns:
            if not IDENT_RE.match(c):
                raise ValueError(f"invalid column identifier: {c!r}")
        return self


class LoadCfg(BaseModel):
    mode: str = Field(default="replace", pattern=r"^(replace|fail_if_exists)$")
    max_rows_per_table: int = Field(default=5_000_000, gt=0)
    max_tables: int = Field(default=30, gt=0)
    # Inferred type must fit at least this share of non-null values; the rest become NULL
    infer_threshold: float = Field(default=0.99, gt=0, le=1)


class SourceCfg(BaseModel):
    type: str = Field(pattern=r"^(csv|parquet|sql|python)$")
    path: str = "raw/"
    options: dict = Field(default_factory=dict)


class DatasetCfg(BaseModel):
    model_config = {"populate_by_name": True}

    name: str = Field(pattern=IDENT)
    schema_: str | None = Field(default=None, alias="schema", pattern=IDENT)
    title: str
    description: str
    source: SourceCfg
    load: LoadCfg = Field(default_factory=LoadCfg)
    tables: dict[str, TableCfg]
    indexes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _checks(self) -> "DatasetCfg":
        self.schema_ = self.schema_ or self.name
        if self.schema_ in RESERVED_SCHEMAS:
            raise ValueError(f"reserved schema name: {self.schema_!r}")
        if not self.tables:
            raise ValueError("dataset must declare at least one table")
        for table_name in self.tables:
            if not IDENT_RE.match(table_name):
                raise ValueError(f"invalid table identifier: {table_name!r}")
        if len(self.tables) > self.load.max_tables:
            raise ValueError(
                f"dataset declares {len(self.tables)} tables, limit is {self.load.max_tables}"
            )
        for table_name, table in self.tables.items():
            for fk in table.foreign_keys:
                if fk.ref_table not in self.tables:
                    raise ValueError(
                        f"table {table_name!r} has a foreign key to unknown table {fk.ref_table!r}"
                    )
        for idx in self.indexes:
            table, cols = parse_table_ref(idx)
            if table not in self.tables:
                raise ValueError(f"index {idx!r} is on unknown table {table!r}")
            if not cols or not all(IDENT_RE.match(c) for c in cols):
                raise ValueError(f"invalid column identifier in index {idx!r}")
        return self

    @property
    def parsed_indexes(self) -> list[tuple[str, list[str]]]:
        return [parse_table_ref(i) for i in self.indexes]

    @property
    def schema_name(self) -> str:
        assert self.schema_ is not None  # set by _checks
        return self.schema_


def load_dataset_cfg(path: str | Path) -> DatasetCfg:
    """Read and validate a dataset.yaml file. `path` may be the file itself or its dataset folder."""
    p = Path(path)
    if p.is_dir():
        p = p / "dataset.yaml"
    if not p.exists():
        raise FileNotFoundError(f"dataset config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{p}: dataset.yaml must be a mapping, got {type(raw).__name__}")
    return DatasetCfg.model_validate(raw)
