import pytest
from pydantic import ValidationError

from ask_data.datasets.config import DatasetCfg, load_dataset_cfg

MINIMAL = {
    "name": "sales",
    "title": "Retail Sales",
    "description": "Orders and customers of an online store.",
    "source": {"type": "csv", "path": "raw/"},
    "tables": {
        "customers": {"primary_key": ["customer_id"]},
        "orders": {
            "primary_key": ["order_id"],
            "foreign_keys": [{"columns": ["customer_id"], "references": "customers(customer_id)"}],
        },
    },
}


def _cfg(**overrides) -> dict:
    merged = {**MINIMAL, **overrides}
    return merged


def test_minimal_config_parses_and_defaults_schema_to_name():
    cfg = DatasetCfg.model_validate(MINIMAL)
    assert cfg.name == "sales"
    assert cfg.schema_name == "sales"


def test_explicit_schema_override_is_respected():
    cfg = DatasetCfg.model_validate(_cfg(schema="sales_v2"))
    assert cfg.schema_name == "sales_v2"


@pytest.mark.parametrize("reserved", ["app", "public", "pg_catalog", "information_schema"])
def test_reserved_schema_name_is_rejected(reserved):
    with pytest.raises(ValidationError, match="reserved schema"):
        DatasetCfg.model_validate(_cfg(schema=reserved))


@pytest.mark.parametrize("bad_name", ["Sales", "1sales", "sales-data", "a" * 32])
def test_invalid_dataset_name_identifier_is_rejected(bad_name):
    with pytest.raises(ValidationError):
        DatasetCfg.model_validate(_cfg(name=bad_name))


def test_foreign_key_to_unknown_table_is_rejected():
    bad = _cfg(
        tables={
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [
                    {"columns": ["customer_id"], "references": "customers(customer_id)"}
                ],
            },
        }
    )
    with pytest.raises(ValidationError, match="unknown table"):
        DatasetCfg.model_validate(bad)


def test_foreign_key_column_count_mismatch_is_rejected():
    bad = _cfg(
        tables={
            "customers": {"primary_key": ["customer_id"]},
            "orders": {
                "primary_key": ["order_id"],
                "foreign_keys": [
                    {"columns": ["customer_id", "region"], "references": "customers(customer_id)"}
                ],
            },
        }
    )
    with pytest.raises(ValidationError, match="foreign key has 2 column"):
        DatasetCfg.model_validate(bad)


def test_foreign_key_bad_references_format_is_rejected():
    bad = _cfg(
        tables={
            "customers": {"primary_key": ["customer_id"]},
            "orders": {
                "foreign_keys": [{"columns": ["customer_id"], "references": "customers"}],
            },
        }
    )
    with pytest.raises(ValidationError, match="references must look like"):
        DatasetCfg.model_validate(bad)


def test_empty_tables_is_rejected():
    with pytest.raises(ValidationError, match="at least one table"):
        DatasetCfg.model_validate(_cfg(tables={}))


@pytest.mark.parametrize("bad_table", ["Orders", "order-items"])
def test_invalid_table_identifier_is_rejected(bad_table):
    bad = _cfg(tables={bad_table: {"primary_key": ["id"]}})
    with pytest.raises(ValidationError):
        DatasetCfg.model_validate(bad)


def test_invalid_column_type_is_rejected():
    bad = _cfg(
        tables={
            "customers": {
                "primary_key": ["customer_id"],
                "columns": {"customer_id": {"type": "varchar"}},
            },
        }
    )
    with pytest.raises(ValidationError, match="unsupported column type"):
        DatasetCfg.model_validate(bad)


def test_numeric_with_precision_type_is_accepted():
    ok = _cfg(
        tables={
            "customers": {
                "primary_key": ["customer_id"],
                "columns": {"balance": {"type": "numeric(12,2)"}},
            },
        }
    )
    cfg = DatasetCfg.model_validate(ok)
    assert cfg.tables["customers"].columns["balance"].type == "numeric(12,2)"


def test_invalid_source_type_is_rejected():
    with pytest.raises(ValidationError):
        DatasetCfg.model_validate(_cfg(source={"type": "excel", "path": "raw/"}))


def test_load_dataset_cfg_round_trip(tmp_path):
    dataset_dir = tmp_path / "sales"
    dataset_dir.mkdir()
    (dataset_dir / "dataset.yaml").write_text(
        """
name: sales
title: Retail Sales
description: Orders and customers of an online store.
source:
  type: csv
  path: raw/
tables:
  customers:
    primary_key: [customer_id]
    columns:
      city: {comment: "City name as typed by the customer; use ILIKE"}
  orders:
    primary_key: [order_id]
    foreign_keys:
      - {columns: [customer_id], references: customers(customer_id)}
    columns:
      order_date: {type: date}
"""
    )
    cfg = load_dataset_cfg(dataset_dir)
    assert cfg.schema_name == "sales"
    assert cfg.tables["orders"].foreign_keys[0].ref_table == "customers"
    assert cfg.tables["customers"].columns["city"].comment.startswith("City name")

    # Also works when pointed straight at the file
    cfg2 = load_dataset_cfg(dataset_dir / "dataset.yaml")
    assert cfg2.name == cfg.name


def test_load_dataset_cfg_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset_cfg(tmp_path / "does-not-exist")


def test_load_dataset_cfg_rejects_non_mapping_yaml(tmp_path):
    bad = tmp_path / "dataset.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_dataset_cfg(bad)
