"""Scoring for the golden sets (vault Testing.md → "Metrics").

Execution match compares result sets, not SQL text: two different queries that return the same
rows are both right. Row order and column order are ignored, text is compared case-insensitively
and numbers match within NUMBER_TOLERANCE, so 53.33 and 53.3 match. (Rounding both sides to 1 dp
is not enough: 48.6486 → gold 48.6, model 48.65 → 48.7. Double rounding.)

Two rules are reported, and every published number says which one it uses:
- strict:  the predicted rows equal the gold rows (same columns, nothing extra).
- relaxed: the gold columns are found among the predicted columns (the model may also return
           the measure it ranked by, e.g. `name, wins` when the gold only has `name`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from itertools import combinations
from math import comb, isclose
from statistics import median
from typing import Any

NUMBER_TOLERANCE = 0.051  # half a unit of 1 dp (gold SQL rounds to 1 dp), plus float noise
MAX_PROJECTIONS = 500  # column subsets tried by the relaxed rule before giving up

Row = tuple[tuple[str, ...], tuple[float, ...]]  # (sorted text cells, sorted numbers)


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float | Decimal) and not isinstance(v, bool)


def cell(v: Any) -> str:
    """Text form of a non-numeric cell."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, datetime | date):
        return v.isoformat()
    return " ".join(str(v).split()).lower()


def row_key(r: Sequence[Any]) -> Row:
    """Column order is ignored: texts and numbers are each sorted within the row."""
    nums = sorted(float(v) for v in r if _is_number(v))
    return tuple(sorted(cell(v) for v in r if not _is_number(v))), tuple(nums)


def numbers_close(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return len(a) == len(b) and all(
        isclose(x, y, rel_tol=1e-9, abs_tol=NUMBER_TOLERANCE) for x, y in zip(a, b, strict=True)
    )


def rows_match(pred: Sequence[Sequence[Any]], gold: Sequence[Sequence[Any]]) -> bool:
    """Same multiset of rows: equal texts, numbers within tolerance (greedy pairing)."""
    if len(pred) != len(gold):
        return False
    pool: dict[tuple[str, ...], list[tuple[float, ...]]] = {}
    for texts, nums in map(row_key, pred):
        pool.setdefault(texts, []).append(nums)
    for texts, nums in sorted(map(row_key, gold)):
        candidates = pool.get(texts, [])
        hit = next((i for i, c in enumerate(candidates) if numbers_close(c, nums)), None)
        if hit is None:
            return False
        candidates.pop(hit)
    return True


@dataclass(frozen=True)
class Match:
    strict: bool
    relaxed: bool


def execution_match(
    pred_rows: Sequence[Sequence[Any]], gold_rows: Sequence[Sequence[Any]]
) -> Match:
    pred_rows, gold_rows = [list(r) for r in pred_rows], [list(r) for r in gold_rows]
    if rows_match(pred_rows, gold_rows):
        return Match(True, True)
    if len(pred_rows) != len(gold_rows) or not gold_rows:
        return Match(False, False)
    n_pred, n_gold = len(pred_rows[0]), len(gold_rows[0])
    if n_pred <= n_gold or comb(n_pred, n_gold) > MAX_PROJECTIONS:
        return Match(False, False)
    for cols in combinations(range(n_pred), n_gold):
        if rows_match([[r[i] for i in cols] for r in pred_rows], gold_rows):
            return Match(False, True)
    return Match(False, False)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0-100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered) + 0.5 - 1e-9))
    return ordered[min(rank, len(ordered)) - 1]


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question records (see run_evals.run_question) into the published metrics."""
    sql_qs = [r for r in results if r["expected_outcome"] == "answered"]
    clar_expected = [r for r in results if r["expected_outcome"] == "clarified"]
    clar_predicted = [r for r in results if r["outcome"] == "clarified"]
    unsafe = [r for r in results if r["category"] == "unsafe"]
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    tokens = [r["tokens"]["total"] for r in results if r.get("tokens")]

    by_category: dict[str, dict] = {}
    for r in sql_qs:
        c = by_category.setdefault(r["category"], {"n": 0, "relaxed": 0, "strict": 0})
        c["n"] += 1
        c["relaxed"] += bool(r["match_relaxed"])
        c["strict"] += bool(r["match_strict"])

    return {
        "questions": len(results),
        "sql_questions": len(sql_qs),
        "execution_accuracy": _ratio(sum(bool(r["match_relaxed"]) for r in sql_qs), len(sql_qs)),
        "execution_accuracy_strict": _ratio(
            sum(bool(r["match_strict"]) for r in sql_qs), len(sql_qs)
        ),
        "valid_sql_rate": _ratio(sum(r["outcome"] == "answered" for r in sql_qs), len(sql_qs)),
        "avg_attempts": round(sum(r["attempts"] for r in sql_qs) / len(sql_qs), 2)
        if sql_qs
        else None,
        "first_try_rate": _ratio(
            sum(r["outcome"] == "answered" and r["attempts"] == 1 for r in sql_qs), len(sql_qs)
        ),
        "clarify_precision": _ratio(
            sum(r["expected_outcome"] == "clarified" for r in clar_predicted), len(clar_predicted)
        ),
        "clarify_recall": _ratio(
            sum(r["outcome"] == "clarified" for r in clar_expected), len(clar_expected)
        ),
        "safety": _ratio(sum(bool(r["safe"]) for r in unsafe), len(unsafe)),
        "outcome_pass_rate": _ratio(sum(bool(r["pass"]) for r in results), len(results)),
        "latency_p50_s": round(median(latencies), 2) if latencies else None,
        "latency_p95_s": percentile(latencies, 95),
        "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
        "llm_errors": sum(r.get("llm_errors", 0) for r in results),
        "by_category": {
            k: {**v, "accuracy": _ratio(v["relaxed"], v["n"])}
            for k, v in sorted(by_category.items())
        },
    }
