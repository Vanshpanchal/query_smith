"""Run a dataset's golden set through the agent and score it (vault Testing.md → "Eval Harness").

    uv run python evals/run_evals.py --dataset ipl                      # baseline, test split
    uv run python evals/run_evals.py --dataset ipl --attempts 1 --label e1-attempts-1
    uv run python evals/run_evals.py --dataset ipl --no-comments --label e2-no-comments
    uv run python evals/run_evals.py --dataset ipl --fewshots 0 --label e3-fewshots-0
    uv run python evals/run_evals.py --dataset ipl --retrieval --label e4-retrieval
    uv run python evals/run_evals.py --dataset ipl --model groq:<other> --label e5-<other>
    uv run python evals/run_evals.py --dataset ipl --ids q001,q014       # a few questions

The graph runs in-process (same nodes, guard, executor and reader role as the API) with an
in-memory checkpointer, one fresh thread per question. Gold SQL runs on the reader role too.
Writes evals/results/<dataset>_<label>_<timestamp>.json and prints a summary. No LangSmith needed;
--langsmith also uploads the golden set and the scored results as a LangSmith experiment.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # `evals.*` when run as a script

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from ask_data.config import settings
from ask_data.datasets.registry import Dataset
from ask_data.graph.state import per_turn_reset
from ask_data.sql.executor import QueryError
from ask_data.sql.guard import UnsafeSQL, guard
from evals.metrics import execution_match, summarize

RESULTS_DIR = ROOT / "evals" / "results"
log = logging.getLogger("evals")


class ErrorCounter(logging.Handler):
    """Counts warnings from the agent's LLM wrappers (failed structured output, provider errors),
    so a run hurt by rate limits is visible instead of looking like a bad model."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0
        self.rate_limited = False

    def reset(self) -> None:
        self.count, self.rate_limited = 0, False

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("ask_data.llm", "ask_data.graph")):
            self.count += 1
            self.rate_limited |= is_rate_limit(record.getMessage())


def is_rate_limit(text: str | None) -> bool:
    text = (text or "").lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


class SkippedAnswer:
    """Evals score the rows, not the wording: skip the answer LLM call unless --with-answer
    (saves ~20 % of the tokens on a rate-limited free tier). The chart is still inferred."""

    async def ainvoke(self, prompt: str, /):
        from ask_data.graph.models import FinalAnswer

        return FinalAnswer(answer="(answer step skipped in evals)")


def load_golden(dataset_dir: Path, split: str | None, ids: set[str] | None) -> list[dict]:
    items = yaml.safe_load((dataset_dir / "golden.yaml").read_text()) or []
    out = []
    for q in items:
        if split and q.get("split", "test") != split:
            continue
        if ids and q["id"] not in ids:
            continue
        out.append({**q, "expected_outcome": q.get("expected_outcome", "answered")})
    return out


def without_comments(ds: Dataset) -> Dataset:
    """E2: the same schema with every table and column COMMENT removed."""
    info = ds.schema_info
    tables = {
        name: replace(t, comment=None, columns=[replace(c, comment=None) for c in t.columns])
        for name, t in info.tables.items()
    }
    return replace(ds, schema_info=replace(info, tables=tables))


def is_safe(outcome: str, sql: str | None, ds: Dataset) -> bool:
    """Nothing but a guarded SELECT may have run: every non-answer is safe, an answer is safe if its
    SQL still passes the guard."""
    if outcome != "answered":
        return True
    try:
        guard(sql or "", ds.schema, ds.tables, settings.max_rows)
        return True
    except UnsafeSQL:
        return False


async def run_question(graph, deps, ds: Dataset, q: dict, gold_rows, errors: ErrorCounter) -> dict:
    usage = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}, "callbacks": [usage]}
    errors.reset()
    started = time.perf_counter()
    crash = None
    try:
        await graph.ainvoke(per_turn_reset(ds.name, q["question"]), config, context=deps)
    except Exception as e:  # noqa: BLE001 — recorded as outcome "error"
        crash = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - started
    snapshot = await graph.aget_state(config)
    state = snapshot.values or {}
    if crash:
        outcome = "error"
    elif snapshot.interrupts:
        outcome = "clarified"
    else:
        outcome = state.get("outcome") or "error"

    match = None
    if q["expected_outcome"] == "answered" and outcome == "answered" and gold_rows is not None:
        match = execution_match(state.get("rows") or [], gold_rows)
    tokens = {"input": 0, "output": 0, "total": 0}
    for u in usage.usage_metadata.values():
        tokens["input"] += u.get("input_tokens", 0)
        tokens["output"] += u.get("output_tokens", 0)
        tokens["total"] += u.get("total_tokens", 0)

    expected = q["expected_outcome"]
    passed = bool(match and match.relaxed) if expected == "answered" else outcome == expected
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected_outcome": expected,
        "outcome": outcome,
        "pass": passed,
        "match_strict": bool(match and match.strict),
        "match_relaxed": bool(match and match.relaxed),
        "safe": is_safe(outcome, state.get("sql"), ds),
        "attempts": state.get("attempts", 0),
        "sql": state.get("sql"),
        "gold_sql": q.get("gold_sql"),
        "columns": state.get("columns", []),
        "pred_rows": state.get("row_count", 0),
        "gold_rows": len(gold_rows) if gold_rows is not None else None,
        "preview": [list(map(str, r)) for r in (state.get("rows") or [])[:5]],
        "expected": q.get("expected"),
        "answer": state.get("answer"),
        "chart": state.get("chart"),
        "error": crash or state.get("error"),
        "standalone_question": state.get("question"),
        "assumption": state.get("assumption"),
        "clarifying_question": state.get("clarifying_question") if outcome == "clarified" else None,
        "tables": state.get("tables", []),
        "table_selection": state.get("table_selection"),
        "latency_s": round(latency, 2),
        "tokens": tokens,
        "llm_errors": errors.count,
        "rate_limited": errors.rate_limited or is_rate_limit(crash or state.get("error")),
    }


def mark(r: dict) -> str:
    if r["pass"]:
        return "✅" if r["match_strict"] or r["expected_outcome"] != "answered" else "🟡"
    return "❌"


def summary_lines(meta: dict, s: dict) -> list[str]:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [
        f"\n{meta['dataset']} · {meta['label']} · {meta['model']} · {s['questions']} questions",
        (
            f"execution accuracy  {pct(s['execution_accuracy'])} relaxed · "
            f"{pct(s['execution_accuracy_strict'])} strict  ({s['sql_questions']} SQL questions)"
        ),
        (
            f"valid SQL {pct(s['valid_sql_rate'])} · first try {pct(s['first_try_rate'])} · "
            f"avg attempts {s['avg_attempts']}"
        ),
        (
            f"clarify P/R {pct(s['clarify_precision'])}/{pct(s['clarify_recall'])} · "
            f"safety {pct(s['safety'])} · outcome pass {pct(s['outcome_pass_rate'])}"
        ),
        (
            f"latency p50 {s['latency_p50_s']} s · p95 {s['latency_p95_s']} s · "
            f"avg tokens {s['avg_tokens']} · LLM warnings {s['llm_errors']}"
        ),
        "by category: "
        + ", ".join(f"{k} {v['relaxed']}/{v['n']}" for k, v in s["by_category"].items()),
        "\nNotes.md row:",
        (
            f"| {meta['started'][:10]} | {meta['commit']} | `{meta['model']}` | {meta['label']}: "
            f"{meta['config_text']} | **{pct(s['execution_accuracy'])}** "
            f"({pct(s['execution_accuracy_strict'])} strict) | {s['avg_attempts']} | "
            f"{s['latency_p95_s']} | |"
        ),
    ]
    return lines


def git_commit() -> str:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                             capture_output=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "datasets", "evals"],
                               cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()  # fmt: skip
        return f"{sha}{'+dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a golden set through the agent and score it.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", help="golden split to run ('' = all)")
    p.add_argument("--ids", help="comma-separated question ids")
    p.add_argument("--limit", type=int, help="first N questions only")
    p.add_argument("--label", default="baseline", help="experiment name in the results file")
    p.add_argument("--model", help="override LLM_MODEL, e.g. groq:<model>")
    p.add_argument("--attempts", type=int, help="override MAX_ATTEMPTS (E1)")
    p.add_argument("--fewshots", type=int, help="few-shot examples in the prompt (E3)")
    p.add_argument("--no-comments", action="store_true", help="drop COMMENT ON text (E2)")
    p.add_argument("--no-notes", action="store_true", help="drop the dataset's notes.md")
    p.add_argument("--retrieval", action="store_true",
                   help="force pgvector schema retrieval for every dataset size (E4)")  # fmt: skip
    p.add_argument("--embedding-model", help="override EMBEDDING_MODEL (e.g. hashing)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    p.add_argument("--retries", type=int, default=6, help="provider retries on 429/5xx")
    p.add_argument("--with-answer", action="store_true", help="also run the answer LLM")
    p.add_argument("--resume", type=Path, help="continue a partial results file (same options)")
    p.add_argument("--langsmith", action="store_true", help="also upload to LangSmith")
    return p.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> dict:
    """Mutate the settings singleton (the nodes read it at call time); return the config used."""
    if args.model:
        settings.llm_model = args.model
    if args.attempts is not None:
        settings.max_attempts = args.attempts
    if args.fewshots is not None:
        settings.fewshots_k = args.fewshots
    if args.embedding_model:
        settings.embedding_model = args.embedding_model
    if args.retrieval:
        settings.schema_retrieval_min_tables = 0
    settings.llm_max_retries = args.retries
    settings.langsmith_tracing = settings.langsmith_tracing and args.langsmith
    return {
        "max_attempts": settings.max_attempts,
        "fewshots_k": settings.fewshots_k,
        "comments": not args.no_comments,
        "notes": not args.no_notes,
        "retrieval": "forced" if args.retrieval else f"> {settings.schema_retrieval_min_tables}",
        "embedding_model": settings.embedding_model if args.retrieval else None,
        "split": args.split or "all",
        "answer_llm": "real" if args.with_answer else "skipped",
    }


async def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    saved_settings = settings.model_dump()  # several runs can share a process (experiments.py)
    try:
        return await _run(args, argv)
    finally:
        for name, value in saved_settings.items():
            setattr(settings, name, value)


async def _run(args: argparse.Namespace, argv) -> int:
    config = apply_overrides(args)

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from ask_data.datasets.registry import DatasetRegistry
    from ask_data.graph.builder import build_graph
    from ask_data.graph.nodes import AgentDeps
    from ask_data.llm import build_structured_llms_or_unavailable, export_provider_env
    from ask_data.schema.retrieval import PgVectorRetriever, build_embedder
    from ask_data.sql.executor import SqlExecutor, reset_session

    export_provider_env()
    llms, status = build_structured_llms_or_unavailable()
    if status != "ok":
        print(f"✗ {status}", file=sys.stderr)
        return 2

    app_pool = AsyncConnectionPool(settings.app_database_url, open=False, min_size=1, max_size=2,
                                   kwargs={"autocommit": True, "row_factory": dict_row})  # fmt: skip
    reader_pool = AsyncConnectionPool(settings.reader_database_url, open=False, min_size=1, max_size=2,
                                      kwargs={"autocommit": True}, reset=reset_session)  # fmt: skip
    await app_pool.open(wait=True, timeout=15)
    await reader_pool.open(wait=True, timeout=15)
    errors = ErrorCounter()
    try:
        async with app_pool.connection() as conn:
            registry = await DatasetRegistry.load(settings.datasets_dir, conn)
        if args.dataset not in registry:
            print(
                f"✗ dataset {args.dataset!r} is not loaded (make load DATASET=...)", file=sys.stderr
            )
            return 2
        ds = registry.get(args.dataset)
        if args.no_comments:
            ds = without_comments(ds)
        if args.no_notes:
            ds = replace(ds, notes="")
        registry._datasets[ds.name] = ds  # the nodes read the dataset from the registry

        retriever = None
        if args.retrieval:
            embedder, emb_status = build_embedder()
            if embedder is None:
                print(f"✗ --retrieval needs an embedder: {emb_status}", file=sys.stderr)
                return 2
            retriever = PgVectorRetriever(app_pool, embedder)

        executor = SqlExecutor(reader_pool, settings.statement_timeout_ms)
        answer_llm = llms["answer"] if args.with_answer else SkippedAnswer()
        deps = AgentDeps(registry=registry, executor=executor, intent_llm=llms["intent"],
                         sql_llm=llms["sql"], answer_llm=answer_llm,
                         rewrite_llm=llms["rewrite"], retriever=retriever)  # fmt: skip
        graph = build_graph(InMemorySaver())

        ids = {i.strip() for i in args.ids.split(",")} if args.ids else None
        golden = load_golden(ROOT / settings.datasets_dir / ds.name, args.split or None, ids)
        golden = golden[: args.limit] if args.limit else golden
        if not golden:
            print("✗ no golden questions selected", file=sys.stderr)
            return 2

        logging.getLogger().addHandler(errors)
        started = datetime.now(UTC)
        meta = {
            "dataset": ds.name,
            "dataset_version": ds.version,
            "schema": ds.schema,
            "label": args.label,
            "model": settings.llm_model,
            "commit": git_commit(),
            "started": started.isoformat(timespec="seconds"),
            "config": config,
            "config_text": ", ".join(f"{k}={v}" for k, v in config.items() if v is not None),
            "match_rule": "relaxed = gold columns ⊆ predicted columns; numbers within ±0.051",
        }
        results: list[dict] = []
        out = RESULTS_DIR / f"{ds.name}_{args.label}_{started.strftime('%Y%m%d_%H%M%S')}.json"
        if args.resume:
            saved = json.loads(args.resume.read_text())
            if saved["meta"]["config"] != config or saved["meta"]["model"] != settings.llm_model:
                print("✗ --resume: options differ from the saved run", file=sys.stderr)
                return 2
            meta, results, out = saved["meta"], saved["results"], args.resume
        done = {r["id"] for r in results}
        todo = [q for q in golden if q["id"] not in done]
        print(f"{ds.name} · {meta['label']} · {settings.llm_model} · {len(golden)} questions"
              f"{f' ({len(done)} already done)' if done else ''}")  # fmt: skip
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        def save(partial: bool) -> dict:
            summary = summarize(results)
            meta["partial"] = partial
            meta["finished"] = None if partial else datetime.now(UTC).isoformat(timespec="seconds")
            out.write_text(json.dumps({"meta": meta, "summary": summary, "results": results},
                                      indent=2, default=str))  # fmt: skip
            return summary

        for i, q in enumerate(todo):
            gold_rows = None
            if q.get("gold_sql"):
                try:
                    res = await executor.run_select(q["gold_sql"], ds.schema, settings.max_rows)
                    gold_rows = res.rows
                except QueryError as e:
                    print(f"⚠ {q['id']}: gold SQL failed: {e}")
            r = await run_question(graph, deps, ds, q, gold_rows, errors)
            if r["rate_limited"]:
                # Not a model result: stop, keep what is done, continue later with --resume
                save(partial=True)
                print(f"\n⏸ {q['id']}: provider rate limit ({(r['error'] or '')[:160]})")
                print(f"  {len(results)}/{len(golden)} done. Continue later with:\n"
                      f"  uv run python evals/run_evals.py {' '.join(sys.argv[1:] if argv is None else argv)}"
                      f" --resume {out.relative_to(ROOT)}")  # fmt: skip
                return 3
            results.append(r)
            save(partial=True)
            print(f"{mark(r)} {r['id']} [{r['category']:<11}] {r['outcome']:<9} "
                  f"a={r['attempts']} {r['latency_s']:5.1f}s  {r['question']}")  # fmt: skip
            if not r["pass"] and r["expected_outcome"] == "answered":
                print(f"     SQL: {' '.join((r['sql'] or '-').split())[:300]}")
                if r["error"]:
                    print(f"     error: {r['error'][:200]}")
            if args.delay and i < len(todo) - 1:
                await asyncio.sleep(args.delay)

        summary = save(partial=False)
        print("\n".join(summary_lines(meta, summary)))
        print(f"\nresults: {out.relative_to(ROOT)}")

        if args.langsmith:
            from evals.langsmith_sync import upload_experiment

            url = await upload_experiment(ds.name, golden, results, meta)
            print(f"LangSmith experiment: {url}")
        return 0
    finally:
        logging.getLogger().removeHandler(errors)
        await reader_pool.close()
        await app_pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
