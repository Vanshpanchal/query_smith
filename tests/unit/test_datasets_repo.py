"""Checks on the datasets shipped in datasets/ (no database, no download needed).

- `sales` is the proof that the loader is generic (ADR-009): its messy CSVs must load from
  dataset.yaml alone. If a loader change breaks it, this fails before anyone runs `make load`.
- Every `golden.yaml` must be well-formed and its gold SQL must only touch its own dataset's
  tables, so a typo is caught here rather than as a wrong eval number in Phase 4.
"""

from pathlib import Path

import pytest
import sqlglot
import yaml
from sqlglot import exp

from ask_data.datasets.config import load_dataset_cfg
from ask_data.datasets.loader import prepare_tables, read_tables

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
GOLDEN_FILES = sorted(DATASETS_DIR.glob("*/golden.yaml"))
CATEGORIES = {
    "lookup", "aggregation", "joins", "time", "window/rank", "domain", "ambiguous", "unsafe",
}  # fmt: skip


def test_sales_loads_from_yaml_alone():
    cfg = load_dataset_cfg(DATASETS_DIR / "sales")
    tables = prepare_tables(cfg, read_tables(cfg, DATASETS_DIR / "sales"))

    types = {(t.name, c.name): c.pg_type for t in tables.values() for c in t.columns}
    inferred = {(t.name, c.name) for t in tables.values() for c in t.columns if c.inferred}
    # messy headers were sanitized to the names dataset.yaml uses
    assert types[("products", "price_inr")] == "numeric(12,2)"  # "Price (INR)", "1,299.00"
    # types the loader had to infer, not read from the yaml
    assert types[("customers", "signup_date")] == "date"  # 27/09/2024, date_dayfirst
    assert types[("customers", "is_premium")] == "boolean"  # yes / no
    assert types[("customers", "pincode")] == "text"  # leading zeros survive
    assert types[("orders", "order_date")] == "date"
    assert {("customers", "signup_date"), ("customers", "pincode"), ("orders", "order_date")} <= (
        inferred
    )
    assert sum(t.row_count for t in tables.values()) > 1000
    assert not any(c.nulled for t in tables.values() for c in t.columns)


@pytest.mark.parametrize("path", GOLDEN_FILES, ids=lambda p: p.parent.name)
def test_golden_file_is_well_formed(path: Path):
    cfg = load_dataset_cfg(path.parent)
    questions = yaml.safe_load(path.read_text())
    assert isinstance(questions, list) and questions

    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate ids"
    texts = [q["question"].strip().lower() for q in questions]
    assert len(texts) == len(set(texts)), "duplicate questions"

    for q in questions:
        assert q["category"] in CATEGORIES, q["id"]
        assert q["split"] in {"train", "test"}, q["id"]
        if q["category"] in {"ambiguous", "unsafe"}:
            assert q.get("expected_outcome") in {"clarified", "declined"}, q["id"]
            continue
        _check_gold_sql(q["id"], q["gold_sql"], cfg.schema_, set(cfg.tables))


def _check_gold_sql(qid: str, gold_sql: str, schema: str, tables: set[str]) -> None:
    statements = sqlglot.parse(gold_sql, read="postgres")
    assert len(statements) == 1, f"{qid}: one statement expected"
    tree = statements[0]
    assert isinstance(tree, exp.Query), f"{qid}: not a SELECT"

    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.name in ctes and not table.db:
            continue
        assert table.db == schema, f"{qid}: {table.sql()} is not qualified with {schema!r}"
        assert table.name in tables, f"{qid}: unknown table {table.name!r}"
