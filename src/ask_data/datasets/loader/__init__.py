"""Generic dataset loader: datasets/<name>/ (dataset.yaml + raw source) -> one Postgres schema.

    cfg     = load_dataset_cfg(dataset_dir)          # validate yaml     (config.py)
    frames  = read_tables(cfg, dataset_dir)          # csv/parquet/python (sources.py)
    tables  = prepare_tables(cfg, frames)            # names, types, keys (prepare.py)
    result  = write_dataset(conn, cfg, tables, ...)  # one transaction    (db.py)

scripts/load_dataset.py is the CLI around this.
"""

from ask_data.datasets.loader.db import LoadResult, write_dataset
from ask_data.datasets.loader.prepare import LoadError, PreparedTable, prepare_tables
from ask_data.datasets.loader.sources import SourceError, dataset_version, read_tables

__all__ = [
    "LoadError",
    "LoadResult",
    "PreparedTable",
    "SourceError",
    "dataset_version",
    "prepare_tables",
    "read_tables",
    "write_dataset",
]
