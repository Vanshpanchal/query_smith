"""Read a dataset's raw source into one DataFrame per table declared in dataset.yaml."""

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from ask_data.datasets.config import DatasetCfg

DEFAULT_NULL_VALUES = frozenset({"", "NULL", "null", "NaN", "nan", "N/A", "n/a", "None"})
# Options we interpret ourselves; everything else in source.options goes to the pandas reader
LOADER_OPTIONS = {"date_dayfirst", "thousands", "null_values"}


class SourceError(ValueError):
    pass


def source_dir(cfg: DatasetCfg, dataset_dir: Path) -> Path:
    return (dataset_dir / cfg.source.path).resolve()


def null_values(cfg: DatasetCfg) -> frozenset[str]:
    extra = cfg.source.options.get("null_values")
    return DEFAULT_NULL_VALUES | frozenset(extra or [])


def read_tables(cfg: DatasetCfg, dataset_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = source_dir(cfg, dataset_dir)
    kind = cfg.source.type
    if kind == "python":
        frames = _run_transform(dataset_dir, raw_dir)
    elif kind in {"csv", "parquet"}:
        if not raw_dir.is_dir():
            raise SourceError(f"source path not found: {raw_dir}")
        frames = {name: _read_file(cfg, raw_dir, name) for name in cfg.tables}
    else:  # "sql": run a dump inside the schema — not needed by any dataset yet
        raise SourceError(f"source type {kind!r} is not supported by the loader yet")

    missing = set(cfg.tables) - set(frames)
    extra = set(frames) - set(cfg.tables)
    if missing or extra:
        raise SourceError(
            f"source tables do not match dataset.yaml: missing={sorted(missing)} "
            f"unexpected={sorted(extra)}"
        )
    for name, df in frames.items():
        if len(df) > cfg.load.max_rows_per_table:
            raise SourceError(
                f"table {name!r} has {len(df):,} rows, limit is {cfg.load.max_rows_per_table:,}"
            )
    return frames


def _read_file(cfg: DatasetCfg, raw_dir: Path, table: str) -> pd.DataFrame:
    ext = cfg.source.type  # csv | parquet
    path = raw_dir / (cfg.tables[table].file or f"{table}.{ext}")
    if not path.is_file():
        raise SourceError(f"table {table!r}: file not found: {path}")
    if ext == "parquet":
        return pd.read_parquet(path)
    reader_opts = {k: v for k, v in cfg.source.options.items() if k not in LOADER_OPTIONS}
    # Everything as text: type inference is ours, so pandas must not guess (or drop zeros)
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=sorted(null_values(cfg)),
        **reader_opts,
    )


def _run_transform(dataset_dir: Path, raw_dir: Path) -> dict[str, pd.DataFrame]:
    path = dataset_dir / "transform.py"
    if not path.is_file():
        raise SourceError(f"source type 'python' needs {path}")
    spec = importlib.util.spec_from_file_location(f"_dataset_transform_{dataset_dir.name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "extract", None)):
        raise SourceError(f"{path} must define extract(raw_dir) -> dict[str, DataFrame]")
    frames = module.extract(raw_dir)
    if not isinstance(frames, dict) or not all(
        isinstance(df, pd.DataFrame) for df in frames.values()
    ):
        raise SourceError(f"{path}: extract() must return dict[str, pandas.DataFrame]")
    return frames


def dataset_version(cfg: DatasetCfg, dataset_dir: Path) -> str:
    """sha256 over dataset.yaml, transform.py (if any) and every file under the source path.

    Stored in app.datasets.version and used in answer-cache keys, so any change to the data
    or its config invalidates cached answers.
    """
    h = hashlib.sha256()
    dataset_dir = dataset_dir.resolve()  # source_dir() is absolute; relative_to needs both
    files = [dataset_dir / "dataset.yaml", dataset_dir / "transform.py"]
    raw_dir = source_dir(cfg, dataset_dir)
    if raw_dir.is_dir():
        files += sorted(p for p in raw_dir.rglob("*") if p.is_file())
    for p in files:
        if not p.is_file():
            continue
        h.update(str(p.relative_to(dataset_dir)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
