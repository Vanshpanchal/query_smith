import pandas as pd
import pytest

from ask_data.datasets.config import DatasetCfg
from ask_data.datasets.loader import LoadError, prepare_tables


def _cfg(tables: dict, indexes: list[str] | None = None, **source_options) -> DatasetCfg:
    return DatasetCfg.model_validate(
        {
            "name": "shop",
            "title": "Shop",
            "description": "Test data.",
            "source": {"type": "csv", "options": source_options},
            "tables": tables,
            "indexes": indexes or [],
        }
    )


SHOP_TABLES = {
    "customers": {"primary_key": ["customer_id"]},
    "orders": {
        "primary_key": ["order_id"],
        "foreign_keys": [{"columns": ["customer_id"], "references": "customers(customer_id)"}],
    },
}


def _frames(customers=None, orders=None) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.DataFrame(
            customers or {"Customer ID": ["1", "2"], "City": ["Pune", "Delhi"]}, dtype=str
        ),
        "orders": pd.DataFrame(
            orders or {"Order ID": ["10", "11"], "CustomerID": ["1", "2"]}, dtype=str
        ),
    }


def _problems(cfg, frames) -> list[str]:
    with pytest.raises(LoadError) as exc:
        prepare_tables(cfg, frames)
    return exc.value.problems


def test_happy_path_types_and_renames():
    tables = prepare_tables(_cfg(SHOP_TABLES), _frames())
    customers = tables["customers"]
    assert customers.column_names() == ["customer_id", "city"]
    assert customers.renamed == {"Customer ID": "customer_id", "City": "city"}
    assert customers.column("customer_id").pg_type == "bigint"
    assert tables["orders"].rows(["order_id", "customer_id"]) == [(10, 1), (11, 2)]


def test_yaml_can_reference_sanitized_column_names():
    tables = {
        "customers": {
            "primary_key": ["customer_id"],
            "columns": {"customer_id": {"type": "int", "comment": "id"}},
        }
    }
    frames = {"customers": pd.DataFrame({"Customer ID": ["1"]}, dtype=str)}
    prepared = prepare_tables(_cfg(tables), frames)
    col = prepared["customers"].column("customer_id")
    assert (col.pg_type, col.inferred) == ("int", False)


def test_unknown_yaml_column_is_reported_with_available_names():
    tables = {"customers": {"primary_key": ["id"], "columns": {"cty": {"comment": "typo"}}}}
    frames = {"customers": pd.DataFrame({"id": ["1"], "city": ["Pune"]}, dtype=str)}
    (problem,) = _problems(_cfg(tables), frames)
    assert "'cty' (columns:)" in problem
    assert "available: id, city" in problem


def test_index_on_missing_column_is_reported():
    tables = {"customers": {"primary_key": ["id"]}}
    frames = {"customers": pd.DataFrame({"id": ["1"]}, dtype=str)}
    (problem,) = _problems(_cfg(tables, indexes=["customers(city)"]), frames)
    assert "'city' (indexes)" in problem


def test_duplicate_and_null_primary_keys_are_reported():
    frames = _frames(customers={"Customer ID": ["1", "1", None, "2"], "City": ["a", "b", "c", "d"]})
    problems = _problems(_cfg(SHOP_TABLES), frames)
    assert any("1 row(s) have NULL in primary key" in p for p in problems)
    assert any("1 duplicate primary key value(s)" in p and "e.g. 1" in p for p in problems)


def test_composite_primary_key_duplicates_show_the_tuple():
    tables = {"items": {"primary_key": ["order_id", "line_no"]}}
    frames = {"items": pd.DataFrame({"order_id": ["1", "1", "1"], "line_no": ["1", "2", "1"]})}
    (problem,) = _problems(_cfg(tables), frames)
    assert "(1, 1)" in problem


def test_fk_orphans_are_counted_with_examples():
    frames = _frames(orders={"Order ID": ["10", "11", "12"], "CustomerID": ["1", "99", "99"]})
    (problem,) = _problems(_cfg(SHOP_TABLES), frames)
    assert problem.startswith("orders(customer_id) -> customers(customer_id): 2 row(s)")
    assert "e.g. 99" in problem
    assert "remove the foreign key" in problem


def test_fk_null_is_not_an_orphan():
    frames = _frames(orders={"Order ID": ["10", "11"], "CustomerID": ["1", None]})
    assert prepare_tables(_cfg(SHOP_TABLES), frames)["orders"].row_count == 2


def test_fk_must_target_the_primary_key():
    tables = {
        "customers": {"primary_key": ["customer_id"]},
        "orders": {
            "primary_key": ["order_id"],
            "foreign_keys": [{"columns": ["city"], "references": "customers(city)"}],
        },
    }
    frames = _frames(orders={"Order ID": ["10"], "City": ["Pune"]})
    (problem,) = _problems(_cfg(tables), frames)
    assert "must reference the primary key of customers" in problem


def test_fk_type_family_mismatch_is_reported():
    # customer ids are codes with leading zeros (text) on one side, numbers on the other
    frames = _frames(
        customers={"Customer ID": ["001", "002"]},
        orders={"Order ID": ["10"], "CustomerID": ["1"]},
    )
    (problem,) = _problems(_cfg(SHOP_TABLES), frames)
    assert "column types differ (customer_id bigint vs customer_id text)" in problem


def test_all_problems_are_reported_together():
    tables = {
        "customers": {"primary_key": ["id"], "columns": {"amount": {"type": "int"}}},
        "orders": {"primary_key": ["id"], "columns": {"missing": {}}},
    }
    frames = {
        "customers": pd.DataFrame({"id": ["1"], "amount": ["abc"]}),
        "orders": pd.DataFrame({"id": ["1"]}),
    }
    problems = _problems(_cfg(tables), frames)
    assert len(problems) == 2
    assert any("not valid int" in p for p in problems)
    assert any("'missing'" in p for p in problems)


def test_source_options_reach_type_parsing():
    tables = {"t": {"primary_key": ["id"]}}
    frames = {"t": pd.DataFrame({"id": ["1"], "d": ["27/09/2026"], "n": ["1,200"]})}
    t = prepare_tables(_cfg(tables, date_dayfirst=True, thousands=","), frames)["t"]
    assert str(t.column("d").values[0]) == "2026-09-27"
    assert t.column("n").values == [1200]


def test_custom_null_values_from_options():
    tables = {"t": {"primary_key": ["id"]}}
    frames = {"t": pd.DataFrame({"id": ["1", "2"], "x": ["5", "-"]})}
    t = prepare_tables(_cfg(tables, null_values=["-"]), frames)["t"]
    assert (t.column("x").pg_type, t.column("x").nulled) == ("bigint", 0)
