"""Graph nodes. Each takes the state (+ runtime context with the services) and returns only the
keys it changes. Nothing here names a dataset: everything domain-specific comes from the
registry entry of `state["dataset"]`.
"""

import logging
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ask_data.config import settings
from ask_data.datasets.registry import DatasetRegistry
from ask_data.graph import prompts
from ask_data.graph.models import FinalAnswer, Intent, SqlDraft
from ask_data.graph.state import AskState
from ask_data.llm import StructuredLLM, ainvoke_structured
from ask_data.sql.executor import Executor, QueryError
from ask_data.sql.guard import UnsafeSQL, guard

log = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Services the nodes need, passed as LangGraph runtime context (`context=` on each call).

    Tests swap the LLMs and the executor for stubs; nothing else changes.
    """

    registry: DatasetRegistry
    executor: Executor
    intent_llm: StructuredLLM
    sql_llm: StructuredLLM
    answer_llm: StructuredLLM


async def classify(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    prompt = prompts.CLASSIFY_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        tables=", ".join(sorted(ds.tables)),
        question=state["question"],
    )
    try:
        intent: Intent = await ainvoke_structured(runtime.context.intent_llm, prompt)
    except Exception as e:  # noqa: BLE001
        # Fail open to "sql": the guard and the DB role are the security layers, not this node
        log.warning("classify failed, defaulting to sql: %s", e)
        return {"intent": "sql"}
    return {"intent": intent.intent}


async def decline(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    examples = "; ".join(f'"{f.question}"' for f in ds.fewshots[:3]) or f'"{ds.title} statistics"'
    text = prompts.DECLINE_MESSAGE.format(
        dataset_title=ds.title, description=ds.description, examples=examples
    )
    return {"answer": text, "outcome": "declined", "messages": [AIMessage(text)]}


async def select_tables(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    # Phase 2: the full schema (every dataset so far has <= SCHEMA_RETRIEVAL_MIN_TABLES tables).
    # Phase 3: pgvector top-k over app.schema_cards + FK expansion for larger datasets.
    ds = runtime.context.registry.get(state["dataset"])
    tables = sorted(ds.tables)
    return {"tables": tables, "schema_context": ds.schema_context()}


async def generate_sql(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    prompt = prompts.SQL_SYSTEM_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        schema=state.get("schema_context") or ds.schema_context(),
        schema_name=ds.schema,
        domain_notes=ds.notes or "(none)",
        fewshots=prompts.format_fewshots(ds.similar_fewshots(state["question"], k=3)),
        max_rows=settings.max_rows,
    )
    prompt += f"\n\nQuestion: {state['question']}"
    if state.get("error"):
        prompt += prompts.REPAIR_PROMPT.format(
            sql=state.get("sql") or "(none)", error=state["error"]
        )

    attempts = state.get("attempts", 0) + 1
    try:
        draft: SqlDraft = await ainvoke_structured(runtime.context.sql_llm, prompt)
    except Exception as e:  # noqa: BLE001 — routed to retry / give_up like any other failure
        return {
            "attempts": attempts,
            "error": f"The model did not return a valid SQL draft: {e}",
            "error_stage": "generate",
        }
    return {
        "sql": draft.sql.strip().rstrip(";").strip(),
        "reasoning": draft.reasoning,
        "attempts": attempts,
        "error": None,
        "error_stage": None,
    }


async def validate_sql(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    try:
        query = guard(state["sql"] or "", ds.schema, ds.tables, settings.max_rows)
        cost = await runtime.context.executor.explain_cost(query, ds.schema)
    except (UnsafeSQL, QueryError) as e:
        return {"error": str(e), "error_stage": "validate"}
    if cost > settings.max_plan_cost:
        return {
            "error": (
                f"Query plan cost {cost:,.0f} exceeds the limit of {settings.max_plan_cost:,.0f}. "
                "Filter earlier, aggregate before joining, and make sure every join has a condition."
            ),
            "error_stage": "validate",
        }
    return {"sql": query, "error": None, "error_stage": None}


async def execute_sql(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    try:
        result = await runtime.context.executor.run_select(
            state["sql"], ds.schema, settings.max_rows
        )
    except QueryError as e:
        return {
            "error": str(e),
            "error_stage": "execute",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }
    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": len(result.rows),
        "truncated": result.truncated,
        "error": None,
        "error_stage": None,
    }


async def check_result(state: AskState) -> dict:
    # An empty result is often a wrong filter value (casing, format). Retry once with a hint,
    # only if attempts remain; otherwise accept "no matching data".
    empty = state.get("row_count", 0) == 0 and not state.get("empty_retried")
    if empty and state.get("attempts", 0) < settings.max_attempts:
        return {
            "error": (
                "The query ran but returned no rows. Check filters, text casing (use ILIKE), "
                "value formats and codes against the column comments. If no rows is truly "
                "the right answer, return the same query."
            ),
            "error_stage": "check",
            "empty_retried": True,
        }
    return {"error": None, "error_stage": None}


async def answer(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    rows = state.get("rows") or []
    preview = "\n".join(str(tuple(r)) for r in rows[: settings.answer_rows])
    prompt = prompts.ANSWER_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        question=state["question"],
        sql=state.get("sql"),
        row_count=state.get("row_count", len(rows)),
        truncated=" (truncated)" if state.get("truncated") else "",
        columns=state.get("columns", []),
        rows=preview or "(no rows)",
    )
    try:
        result: FinalAnswer = await ainvoke_structured(runtime.context.answer_llm, prompt)
        text = result.answer
    except Exception as e:  # noqa: BLE001 — the rows are still worth returning
        log.warning("answer LLM failed: %s", e)
        text = f"The query returned {len(rows)} row(s); see the table for the result."
    return {
        "answer": text,
        "chart": None,  # Phase 4: ChartSpec from the answer node
        "outcome": "answered",
        "last_sql": state.get("sql"),
        "messages": [AIMessage(text)],
    }


async def give_up(state: AskState) -> dict:
    text = prompts.GIVE_UP_MESSAGE.format(
        attempts=state.get("attempts", 0), error=state.get("error") or "unknown error"
    )
    if state.get("sql"):
        text += f"\n\nLast SQL tried:\n{state['sql']}"
    return {"answer": text, "outcome": "gave_up", "messages": [AIMessage(text)]}
