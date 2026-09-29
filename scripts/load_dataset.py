"""Load datasets/<name>/ into its own Postgres schema.

    uv run python scripts/load_dataset.py sales            # validate + load
    uv run python scripts/load_dataset.py sales --dry-run  # validate + show inferred types only

Runs as askdata_app (APP_DATABASE_URL) and grants SELECT to the role in READER_DATABASE_URL.
"""

import argparse
import sys
import time
from pathlib import Path

from ask_data.datasets.config import load_dataset_cfg
from ask_data.datasets.loader import (
    LoadError,
    PreparedTable,
    SourceError,
    dataset_version,
    prepare_tables,
    read_tables,
    write_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="dataset folder name under DATASETS_DIR")
    parser.add_argument("--datasets-dir", help="default: DATASETS_DIR setting (datasets/)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print, no database")
    args = parser.parse_args()

    started = time.perf_counter()
    datasets_dir = Path(args.datasets_dir or _settings().datasets_dir)
    dataset_dir = datasets_dir / args.name
    try:
        cfg = load_dataset_cfg(dataset_dir)
        if cfg.name != args.name:
            raise LoadError([f"dataset.yaml name {cfg.name!r} != folder name {args.name!r}"])
        frames = read_tables(cfg, dataset_dir)
        tables = prepare_tables(cfg, frames)
    except (LoadError, SourceError, FileNotFoundError, ValueError) as e:
        print(f"✗ {args.name}: dataset is not loadable:\n{e}", file=sys.stderr)
        return 1

    for table in tables.values():
        _print_table(table)
    version = dataset_version(cfg, dataset_dir)
    if args.dry_run:
        print(f"\ndry run: {len(tables)} table(s) valid, version {version[:12]}; nothing written")
        return 0

    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    settings = _settings()
    reader_role = conninfo_to_dict(settings.reader_database_url).get("user", "askdata_reader")
    try:
        with psycopg.connect(settings.app_database_url, autocommit=True) as conn:
            result = write_dataset(conn, cfg, tables, version, str(reader_role))
    except (LoadError, psycopg.Error) as e:
        print(f"✗ {args.name}: load failed, previous version left intact:\n{e}", file=sys.stderr)
        return 1

    print(
        f"\n✓ loaded {cfg.name} → schema {result.schema}: {result.table_count} tables, "
        f"{result.row_count:,} rows, version {result.version[:12]} "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return 0


def _print_table(table: PreparedTable) -> None:
    print(f"\n{table.name}  ({table.row_count:,} rows)")
    for c in table.columns:
        source = "inferred" if c.inferred else "yaml"
        note = ""
        if c.nulled:
            examples = ", ".join(repr(v) for v in c.nulled_examples)
            note = f"  ⚠ {c.nulled} value(s) did not fit and became NULL, e.g. {examples}"
        print(f"  {c.name:<32} {c.pg_type:<18} {source}{note}")
    for src, dst in table.renamed.items():
        print(f"  renamed {src!r} → {dst}")


def _settings():
    from ask_data.config import settings

    return settings


if __name__ == "__main__":
    sys.exit(main())
