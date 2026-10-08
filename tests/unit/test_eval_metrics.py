"""evals/metrics.py: the scoring rule every published accuracy number depends on."""

from datetime import date
from decimal import Decimal

import pytest

from evals.metrics import cell, execution_match, percentile, summarize
from evals.run_evals import load_golden, without_comments


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (Decimal("53.33"), 53.3),  # the q012 miss: precision, not a wrong answer
        (Decimal("48.65"), Decimal("48.6")),  # 48.6486: half-up rounding both would give 48.7
        (11, 11.0),
        ("  Gujarat   Titans ", "gujarat titans"),
        (0, Decimal("0.00")),
        (True, "true"),
    ],
)
def test_cells_that_match(a, b):
    assert execution_match([[a]], [[b]]).strict


@pytest.mark.parametrize(("a", "b"), [(53.3, 53.4), (11, 12), ("a", "b"), (None, ""), (1, "1x")])
def test_cells_that_differ(a, b):
    assert not execution_match([[a]], [[b]]).relaxed


def test_dates_compare_as_iso_text():
    assert cell(date(2023, 5, 29)) == "2023-05-29"


def test_order_of_rows_and_columns_is_ignored():
    gold = [("Gujarat Titans", 11), ("Chennai Super Kings", 10)]
    pred = [[10, "Chennai Super Kings"], [11, "Gujarat Titans"]]
    m = execution_match(pred, gold)
    assert m.strict and m.relaxed


def test_extra_column_passes_relaxed_only():
    gold = [("Gujarat Titans",)]
    pred = [["Gujarat Titans", 11]]
    m = execution_match(pred, gold)
    assert not m.strict and m.relaxed


def test_relaxed_still_needs_the_same_rows():
    gold = [("Gujarat Titans",)]
    assert not execution_match([["Chennai Super Kings", 10]], gold).relaxed
    assert not execution_match([["Gujarat Titans", 11], ["CSK", 10]], gold).relaxed  # extra row


def test_relaxed_keeps_rows_aligned():
    # Each column's values appear in the prediction, but paired differently: not a match
    gold = [("a", 1), ("b", 2)]
    pred = [["a", 2, "x"], ["b", 1, "y"]]
    assert not execution_match(pred, gold).relaxed


def test_missing_column_fails():
    assert not execution_match([["Gujarat Titans"]], [("Gujarat Titans", 11)]).relaxed


def test_empty_results():
    assert execution_match([], []).strict
    assert not execution_match([["x"]], []).relaxed


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 95) == 19.0
    assert percentile(values, 50) == 10.0
    assert percentile([], 95) is None


def _record(**kw):
    base = {
        "category": "joins", "expected_outcome": "answered", "outcome": "answered",
        "pass": True, "match_strict": True, "match_relaxed": True, "safe": True,
        "attempts": 1, "latency_s": 1.0, "tokens": {"total": 100}, "llm_errors": 0,
    }  # fmt: skip
    return {**base, **kw}


def test_summarize():
    miss = {"pass": False, "match_strict": False, "match_relaxed": False}
    results = [
        _record(),
        _record(match_strict=False, attempts=2),  # relaxed pass, second try
        _record(**miss, outcome="gave_up", attempts=3, latency_s=9.0),
        _record(category="ambiguous", expected_outcome="clarified", outcome="clarified"),
        _record(**miss, category="window", outcome="clarified", attempts=0),  # wrongly clarified
        _record(category="unsafe", expected_outcome="declined", outcome="declined"),
    ]
    s = summarize(results)
    assert s["sql_questions"] == 4
    assert s["execution_accuracy"] == 0.5 and s["execution_accuracy_strict"] == 0.25
    assert s["valid_sql_rate"] == 0.5
    assert s["clarify_precision"] == 0.5 and s["clarify_recall"] == 1.0
    assert s["safety"] == 1.0
    assert s["by_category"]["joins"] == {"n": 3, "relaxed": 2, "strict": 1, "accuracy": 0.6667}


def test_load_golden_defaults_and_filters(tmp_path):
    (tmp_path / "golden.yaml").write_text(
        "- {id: q1, question: 'a?', gold_sql: SELECT 1, category: lookup, split: test}\n"
        "- {id: q2, question: 'b?', expected_outcome: clarified, category: ambiguous, split: test}\n"
        "- {id: q3, question: 'c?', gold_sql: SELECT 1, category: lookup, split: train}\n"
    )
    qs = load_golden(tmp_path, "test", None)
    assert [q["id"] for q in qs] == ["q1", "q2"]
    assert qs[0]["expected_outcome"] == "answered"
    assert [q["id"] for q in load_golden(tmp_path, None, {"q3"})] == ["q3"]


def test_without_comments_strips_table_and_column_comments():
    from conftest import make_dataset

    ds = make_dataset()
    stripped = without_comments(ds)
    rendered = stripped.schema_context()
    assert "--" not in rendered and "TABLE shop.items" in rendered
    assert "Item name" in ds.schema_context()  # the original is untouched
