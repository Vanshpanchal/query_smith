from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Outcome = Literal["answered", "declined", "gave_up"]
Stage = Literal["generate", "validate", "execute", "check"]


class ChartSpec(TypedDict):
    type: Literal["bar", "line", "pie", "table"]
    x: str | None
    y: str | None


class AskState(TypedDict, total=False):
    dataset: str  # set once when the thread is created; never changes
    messages: Annotated[list[AnyMessage], add_messages]  # persists across turns
    last_sql: str | None  # persists across turns (follow-ups, Phase 3)

    # --- per-turn fields: reset by the API on every new question (PER_TURN_RESET) ---
    question: str  # standalone question (Phase 3: rewritten by `contextualize`)
    intent: Literal["sql", "clarify", "decline"]
    clarifying_question: str | None
    clarify_count: int
    tables: list[str]
    schema_context: str
    reasoning: str | None
    sql: str | None
    attempts: int
    error: str | None
    error_stage: Stage | None  # where the last error came from (SSE `retry` event)
    empty_retried: bool  # check_result retries an empty result only once
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    answer: str | None
    chart: ChartSpec | None
    outcome: Outcome | None


def per_turn_reset(dataset: str, question: str) -> dict:
    """Inputs for a new question on a thread.

    The checkpointer keeps every key between turns, so without this a second question would start
    at attempts=3 and give up at once (Architecture gotcha #1).
    """
    from langchain_core.messages import HumanMessage

    return {
        "dataset": dataset,
        "messages": [HumanMessage(question)],
        "question": question,
        "intent": "sql",
        "clarifying_question": None,
        "clarify_count": 0,
        "tables": [],
        "schema_context": "",
        "reasoning": None,
        "sql": None,
        "attempts": 0,
        "error": None,
        "error_stage": None,
        "empty_retried": False,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "answer": None,
        "chart": None,
        "outcome": None,
    }
