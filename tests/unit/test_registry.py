"""Registry helpers + checks on every dataset's notes.md / fewshots.yaml (no database)."""

from pathlib import Path

import pytest
import sqlglot
import yaml
from conftest import make_dataset

from ask_data.datasets.config import load_dataset_cfg
from ask_data.datasets.registry import Fewshot, read_fewshots, read_notes
from ask_data.graph import prompts
from ask_data.sql.guard import guard

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
FEWSHOT_DIRS = sorted(p.parent for p in DATASETS_DIR.glob("*/fewshots.yaml"))


def test_similar_fewshots_ranks_by_overlap():
    ds = make_dataset()
    ds.fewshots = [
        Fewshot("Total revenue by month", "SELECT 1"),
        Fewshot("Top 5 products by revenue in 2024", "SELECT 2"),
        Fewshot("Orders per city", "SELECT 3"),
    ]
    top = ds.similar_fewshots("top 3 products by revenue", k=2)
    assert top[0].sql == "SELECT 2"
    assert len(top) == 2


def test_similar_fewshots_falls_back_to_first_k():
    ds = make_dataset()
    assert ds.similar_fewshots("zzz qqq", k=1) == ds.fewshots[:1]


def test_schema_context_renders_comments_and_keys():
    text = make_dataset().schema_context()
    assert "TABLE shop.items (" in text and "-- Item name" in text
    assert "FOREIGN KEY (item_id) REFERENCES items(id)" in text


def test_prompt_slots_filled():
    ds = make_dataset()
    text = prompts.SQL_SYSTEM_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        schema=ds.schema_context(),
        schema_name=ds.schema,
        domain_notes=ds.notes,
        fewshots=prompts.format_fewshots(ds.fewshots),
        max_rows=1000,
    )
    assert "{" not in text.replace("{ ", "")  # no unfilled slot
    assert ds.notes in text and "How many items are there?" in text


@pytest.mark.parametrize("folder", FEWSHOT_DIRS, ids=lambda p: p.name)
def test_fewshots_are_safe_and_not_in_golden(folder: Path):
    cfg = load_dataset_cfg(folder)
    shots = read_fewshots(folder)
    assert 3 <= len(shots) <= 8
    for shot in shots:
        # every few-shot passes the same guard the agent's SQL must pass
        guard(shot.sql, cfg.schema_name, set(cfg.tables), 1000)
        assert len(sqlglot.parse(shot.sql, read="postgres")) == 1
    golden = folder / "golden.yaml"
    if golden.exists():
        gold_questions = {q["question"].strip().lower() for q in yaml.safe_load(golden.read_text())}
        overlap = {s.question.lower() for s in shots} & gold_questions
        assert not overlap, f"few-shot questions leak into golden.yaml: {overlap}"


@pytest.mark.parametrize(
    "folder", sorted(p.parent for p in DATASETS_DIR.glob("*/notes.md")), ids=lambda p: p.name
)
def test_notes_are_short(folder: Path):
    notes = read_notes(folder)
    assert notes and len(notes.splitlines()) <= 40


def test_summary_suggests_verified_fewshot_questions():
    from conftest import make_dataset

    summary = make_dataset().summary()
    assert summary["examples"] == ["How many items are there?"]
