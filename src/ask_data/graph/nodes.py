"""Graph nodes. Each takes the state (+ runtime context with the services) and returns only the
keys it changes. Nothing here names a dataset: everything domain-specific comes from the
registry entry of `state["dataset"]`.
"""

import logging
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from ask_data.config import settings
from ask_data.datasets.registry import DatasetRegistry
from ask_data.graph import prompts
from ask_data.graph.charts import infer_chart
from ask_data.graph.models import FinalAnswer, Intent, SqlDraft, StandaloneQuestion
from ask_data.graph.state import AskState
from ask_data.llm import StructuredLLM, ainvoke_structured
from ask_data.schema.retrieval import SchemaRetriever, expand_with_fks
from ask_data.sql.executor import Executor, QueryError
from ask_data.sql.guard import UnsafeSQL, guard

log = logging.getLogger(__name__)

MIN_CLARIFY_WORDS = 4  # shorter is a label ("Best player"), not a question to show a user


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
    rewrite_llm: StructuredLLM
    retriever: SchemaRetriever | None = None  # None → always the full schema


async def contextualize(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    """Rewrite a follow-up ("now only finals") or a clarified question into a standalone one."""
    messages = state.get("messages") or []
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    if len(humans) <= 1:  # first question of the thread, no clarification: nothing to resolve
        return {"question": state["question"]}

    ds = runtime.context.registry.get(state["dataset"])
    prompt = prompts.CONTEXTUALIZE_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        history=prompts.format_history(messages, settings.history_messages),
        last_sql=state.get("last_sql") or "(none)",
    )
    try:
        result: StandaloneQuestion = await ainvoke_structured(runtime.context.rewrite_llm, prompt)
        question = " ".join(result.question.split())
    except Exception as e:  # noqa: BLE001 — the raw question is still worth trying
        log.warning("contextualize failed, using the question as asked: %s", e)
        question = ""
    if not question:
        question = state["question"]
        if state.get("clarification"):
            question += f" ({state['clarification']})"
    return {"question": question}


async def classify(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    can_clarify = state.get("clarify_count", 0) == 0  # one clarifying question per turn (ADR-008)
    prompt = prompts.CLASSIFY_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        tables=", ".join(sorted(ds.tables)),
        clarify_rule=prompts.CLARIFY_ALLOWED if can_clarify else prompts.CLARIFY_USED,
        question=state["question"],
    )
    try:
        intent: Intent = await ainvoke_structured(runtime.context.intent_llm, prompt)
    except Exception as e:  # noqa: BLE001
        # Fail open to "sql": the guard and the DB role are the security layers, not this node
        log.warning("classify failed, defaulting to sql: %s", e)
        return {"intent": "sql", "clarifying_question": None, "assumption": None}

    question = clarifying_question(intent)
    if intent.intent == "clarify" and can_clarify and question:
        return {"intent": "clarify", "clarifying_question": question, "assumption": None}
    if intent.intent == "decline":
        return {"intent": "decline", "clarifying_question": None, "assumption": None}
    # "sql", or a "clarify" that isn't allowed (second time) or isn't a usable question ("Best"):
    # answer with a stated reading instead
    return {
        "intent": "sql",
        "clarifying_question": None,
        "assumption": (intent.assumption or "").strip() or None,
    }


def clarifying_question(intent: Intent) -> str | None:
    """The model's question if it is one; else one built from its readings. Real models
    sometimes put a label ("Best player") in the question field but list the readings well."""
    question = " ".join((intent.clarifying_question or "").split())
    if len(question.split()) >= MIN_CLARIFY_WORDS and question.endswith("?"):
        return question
    readings = [" ".join(r.split()).rstrip("?.") for r in intent.readings if r.strip()][:3]
    if len(readings) >= 2:
        return f"Do you mean {', '.join(readings[:-1])} or {readings[-1]}?"
    return None


async def clarify(state: AskState) -> dict:
    """Pause the run and ask the user. On resume LangGraph re-runs this node from the top and
    interrupt() returns the reply, so nothing before it may have side effects."""
    question = state["clarifying_question"]
    reply = " ".join(str(interrupt({"question": question})).split())
    return {
        "messages": [AIMessage(question), HumanMessage(reply)],
        "clarification": reply,
        "clarify_count": state.get("clarify_count", 0) + 1,
    }


async def decline(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    examples = "; ".join(f'"{f.question}"' for f in ds.fewshots[:3]) or f'"{ds.title} statistics"'
    text = prompts.DECLINE_MESSAGE.format(
        dataset_title=ds.title, description=ds.description, examples=examples
    )
    return {"answer": text, "outcome": "declined", "messages": [AIMessage(text)]}


async def select_tables(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    """Full schema for small datasets; otherwise the nearest table cards (pgvector) + the tables
    needed to join them. Any retrieval problem falls back to the full schema."""
    ds = runtime.context.registry.get(state["dataset"])
    full = {"tables": sorted(ds.tables), "schema_context": ds.schema_context(),
            "table_selection": "full"}  # fmt: skip
    retriever = runtime.context.retriever
    if retriever is None or len(ds.tables) <= settings.schema_retrieval_min_tables:
        return full
    try:
        hits = await retriever.search(ds.name, state["question"], settings.schema_retrieval_top_k)
    except Exception as e:  # noqa: BLE001
        log.warning("schema retrieval failed for %r, sending the full schema: %s", ds.name, e)
        return full
    tables = expand_with_fks(hits, ds.schema_info)
    if not tables:  # dataset not indexed (or indexed with another embedding model)
        log.warning("no schema cards for %r; run scripts/index_schema.py %s", ds.name, ds.name)
        return full
    return {"tables": tables, "schema_context": ds.schema_context(tables),
            "table_selection": "retrieved"}  # fmt: skip


async def generate_sql(state: AskState, runtime: Runtime[AgentDeps]) -> dict:
    ds = runtime.context.registry.get(state["dataset"])
    prompt = prompts.SQL_SYSTEM_PROMPT.format(
        dataset_title=ds.title,
        dataset_description=ds.description,
        schema=state.get("schema_context") or ds.schema_context(),
        schema_name=ds.schema,
        domain_notes=ds.notes or "(none)",
        fewshots=prompts.format_fewshots(
            ds.similar_fewshots(state["question"], k=settings.fewshots_k)
        ),
        max_rows=settings.max_rows,
    )
    if state.get("last_sql"):
        prompt += prompts.PREVIOUS_SQL.format(last_sql=state["last_sql"])
    prompt += f"\n\nQuestion: {state['question']}"
    if state.get("assumption"):
        prompt += prompts.INTERPRETATION.format(assumption=state["assumption"])
    if state.get("error"):
        prompt += prompts.REPAIR_PROMPT.format(
            sql=state.get("sql") or "(none)", error=state["error"]
        )

    attempts = state.get("attempts", 0) + 1
    try:
        draft: SqlDraft = await ainvoke_structured(runtime.context.sql_llm, prompt)
    except Exception as e:  # noqa: BLE001 — routed to retry / give_up like any other failure
        log.warning("generate_sql: no valid draft: %s", e)
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
        assumption_rule=(
            prompts.ASSUMPTION_RULE.format(assumption=state["assumption"])
            if state.get("assumption")
            else ""
        ),
    )
    hint = None
    try:
        result: FinalAnswer = await ainvoke_structured(runtime.context.answer_llm, prompt)
        text, hint = result.answer, result.chart
    except Exception as e:  # noqa: BLE001 — the rows are still worth returning
        log.warning("answer LLM failed: %s", e)
        text = f"The query returned {len(rows)} row(s); see the table for the result."
    return {
        "answer": text,
        "chart": infer_chart(state.get("columns") or [], rows, hint),
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
