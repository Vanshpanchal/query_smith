"""Re-score saved eval results without calling the LLM: re-run each stored predicted SQL and gold
SQL on the reader role and apply the current metrics.py. Use it after a metric change.

    uv run python evals/rescore.py evals/results/ipl_baseline_20261008_170934.json [...]

Rewrites each file in place (keeps the original summary as `summary_before_rescore`).
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from psycopg_pool import AsyncConnectionPool

from ask_data.config import settings
from ask_data.sql.executor import QueryError, SqlExecutor, reset_session
from evals.metrics import execution_match, summarize


async def rescore(path: Path, executor: SqlExecutor) -> None:
    data = json.loads(path.read_text())
    schema = data["meta"].get("schema") or data["meta"]["dataset"]
    changed = 0
    for r in data["results"]:
        if r["expected_outcome"] != "answered" or r["outcome"] != "answered":
            continue
        try:
            pred = (await executor.run_select(r["sql"], schema, settings.max_rows)).rows
            gold = (await executor.run_select(r["gold_sql"], schema, settings.max_rows)).rows
        except QueryError as e:
            print(f"⚠ {r['id']}: {e}")
            continue
        m = execution_match(pred, gold)
        if (m.strict, m.relaxed) != (r["match_strict"], r["match_relaxed"]):
            changed += 1
            print(f"  {r['id']}: relaxed {r['match_relaxed']} → {m.relaxed}, "
                  f"strict {r['match_strict']} → {m.strict}")  # fmt: skip
        r["match_strict"], r["match_relaxed"], r["pass"] = m.strict, m.relaxed, m.relaxed
    data.setdefault("summary_before_rescore", data["summary"])
    data["summary"] = summarize(data["results"])
    data["meta"]["match_rule"] = "relaxed = gold columns ⊆ predicted columns; numbers within ±0.051"
    path.write_text(json.dumps(data, indent=2, default=str))
    s = data["summary"]
    print(f"{path.name}: {changed} changed · accuracy {s['execution_accuracy']:.0%} relaxed, "
          f"{s['execution_accuracy_strict']:.0%} strict")  # fmt: skip


async def main(paths: list[str]) -> int:
    pool = AsyncConnectionPool(settings.reader_database_url, open=False, min_size=1, max_size=2,
                               kwargs={"autocommit": True}, reset=reset_session)  # fmt: skip
    await pool.open(wait=True, timeout=15)
    try:
        executor = SqlExecutor(pool, settings.statement_timeout_ms)
        for p in paths:
            await rescore(Path(p), executor)
    finally:
        await pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
