import os
from pathlib import Path

import pandas as pd
import pytest

from ask_data.datasets.config import load_dataset_cfg
from ask_data.datasets.loader import SourceError, dataset_version, prepare_tables, read_tables

CSV_YAML = """\
name: shop
title: Shop
description: Test data.
source:
  type: csv
  options: {thousands: ","}
tables:
  customers:
    primary_key: [customer_id]
  orders:
    file: order_export.csv
    primary_key: [order_id]
"""

PY_YAML = """\
name: games
title: Games
description: Test data.
source: {type: python}
tables:
  teams: {primary_key: [id]}
"""


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    d = tmp_path / "shop"
    (d / "raw").mkdir(parents=True)
    (d / "dataset.yaml").write_text(CSV_YAML)
    (d / "raw" / "customers.csv").write_text('customer_id,pincode,spend\n1,011001,"1,200"\n2,,5\n')
    (d / "raw" / "order_export.csv").write_text("order_id,customer_id\n10,1\n")
    return d


def test_csv_is_read_as_text_so_leading_zeros_survive(csv_dataset):
    frames = read_tables(load_dataset_cfg(csv_dataset), csv_dataset)
    customers = frames["customers"]
    assert customers["pincode"].tolist()[0] == "011001"
    assert customers["spend"].tolist()[0] == "1,200"
    assert pd.isna(customers["pincode"].tolist()[1])


def test_custom_file_name_is_used(csv_dataset):
    frames = read_tables(load_dataset_cfg(csv_dataset), csv_dataset)
    assert frames["orders"]["order_id"].tolist() == ["10"]


def test_missing_file_is_a_source_error(csv_dataset):
    (csv_dataset / "raw" / "order_export.csv").unlink()
    with pytest.raises(SourceError, match="file not found"):
        read_tables(load_dataset_cfg(csv_dataset), csv_dataset)


def test_max_rows_per_table_is_enforced(csv_dataset):
    (csv_dataset / "dataset.yaml").write_text(CSV_YAML + "load: {max_rows_per_table: 1}\n")
    with pytest.raises(SourceError, match="limit is 1"):
        read_tables(load_dataset_cfg(csv_dataset), csv_dataset)


def test_sql_source_says_it_is_not_supported(csv_dataset):
    (csv_dataset / "dataset.yaml").write_text(CSV_YAML.replace("type: csv", "type: sql"))
    with pytest.raises(SourceError, match="not supported"):
        read_tables(load_dataset_cfg(csv_dataset), csv_dataset)


def test_version_changes_when_data_changes(csv_dataset):
    cfg = load_dataset_cfg(csv_dataset)
    before = dataset_version(cfg, csv_dataset)
    assert dataset_version(cfg, csv_dataset) == before
    (csv_dataset / "raw" / "order_export.csv").write_text("order_id,customer_id\n10,2\n")
    assert dataset_version(cfg, csv_dataset) != before


def test_version_works_with_a_relative_dataset_path(csv_dataset, monkeypatch):
    # Regression (Phase 1.2): `make load` passes the relative datasets/ dir, which crashed
    # relative_to() against the absolute source dir.
    monkeypatch.chdir(csv_dataset.parent)
    rel = Path("shop")
    cfg = load_dataset_cfg(rel)
    assert not rel.is_absolute()
    assert dataset_version(cfg, rel) == dataset_version(cfg, csv_dataset)
    assert set(read_tables(cfg, rel)) == {"customers", "orders"}


# --- python source ------------------------------------------------------------------------------


def _python_dataset(tmp_path: Path, transform: str) -> Path:
    d = tmp_path / "games"
    (d / "raw").mkdir(parents=True)
    (d / "dataset.yaml").write_text(PY_YAML)
    (d / "transform.py").write_text(transform)
    return d


def test_python_source_calls_extract_with_the_raw_dir(tmp_path):
    d = _python_dataset(
        tmp_path,
        "import pandas as pd\n"
        "def extract(raw_dir):\n"
        "    assert raw_dir.name == 'raw' and raw_dir.is_absolute()\n"
        "    return {'teams': pd.DataFrame({'id': [1.0, 2.0], 'name': ['a', 'b']})}\n",
    )
    cfg = load_dataset_cfg(d)
    tables = prepare_tables(cfg, read_tables(cfg, d))
    assert tables["teams"].column("id").pg_type == "bigint"


def test_python_source_without_transform_is_an_error(tmp_path):
    d = _python_dataset(tmp_path, "")
    os.remove(d / "transform.py")
    with pytest.raises(SourceError, match="needs"):
        read_tables(load_dataset_cfg(d), d)


def test_transform_without_extract_is_an_error(tmp_path):
    d = _python_dataset(tmp_path, "x = 1\n")
    with pytest.raises(SourceError, match="must define extract"):
        read_tables(load_dataset_cfg(d), d)


def test_transform_must_return_the_declared_tables(tmp_path):
    d = _python_dataset(
        tmp_path,
        "import pandas as pd\ndef extract(raw_dir):\n    return {'team': pd.DataFrame({'id': [1]})}\n",
    )
    with pytest.raises(SourceError, match=r"missing=\['teams'\] unexpected=\['team'\]"):
        read_tables(load_dataset_cfg(d), d)


def test_transform_must_return_dataframes(tmp_path):
    d = _python_dataset(tmp_path, "def extract(raw_dir):\n    return {'teams': [1, 2]}\n")
    with pytest.raises(SourceError, match="must return"):
        read_tables(load_dataset_cfg(d), d)


def test_version_includes_transform_py(tmp_path):
    d = _python_dataset(tmp_path, "def extract(raw_dir):\n    return {}\n")
    cfg = load_dataset_cfg(d)
    before = dataset_version(cfg, d)
    (d / "transform.py").write_text("def extract(raw_dir):\n    return {}  # changed\n")
    assert dataset_version(cfg, d) != before
