"""Graph run → SSE events (contract: vault API.md → "SSE Event Contract").

`EventMapper` is a pure function of the `stream_mode="updates"` chunks, so it is unit-tested
without a graph. `run_turn` drives the graph and always ends with `done`.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ask_data.config import settings

log = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]

STATUS = {
    "classify": "Understanding the question…",
    "validate_sql": "SQL passed the safety checks",
}
FINAL_NODES = ("answer", "decline", "give_up")
CHECK_STAGE = {"generate_sql": "generate", "validate_sql": "validate",
               "execute_sql": "execute", "check_result": "check"}  # fmt: skip


class EventMapper:
    def __init__(self, query_id: str, stream_rows: int | None = None):
        self.query_id = query_id
        self.stream_rows = stream_rows or settings.stream_rows
        self.attempt = 0
        self.sql: str | None = None
        self.final: dict[str, Any] | None = None

    def feed(self, chunk: dict[str, Any]) -> list[Event]:
        events: list[Event] = []
        for node, update in chunk.items():
            if node == "__interrupt__":
                for intr in update:
                    events.append(("clarify", {"question": _interrupt_question(intr)}))
                continue
            update = update or {}
            if "attempts" in update:
                self.attempt = update["attempts"]
            if update.get("sql"):
                self.sql = update["sql"]
            events.extend(self._map(node, update))
        return events

    def _map(self, node: str, update: dict[str, Any]) -> list[Event]:
        if node in CHECK_STAGE and update.get("error"):
            return [("retry", {"stage": CHECK_STAGE[node], "error": update["error"],
                               "attempt": self.attempt})]  # fmt: skip
        if node in STATUS:
            return [("status", {"node": node, "message": STATUS[node]})]
        if node == "select_tables":
            return [("tables", {"tables": update.get("tables", [])})]
        if node == "generate_sql":
            return [("sql", {"sql": update.get("sql"), "attempt": self.attempt,
                             "reasoning": update.get("reasoning")})]  # fmt: skip
        if node == "execute_sql":
            rows = update.get("rows", [])
            return [("rows", {
                "columns": update.get("columns", []),
                "rows": rows[: self.stream_rows],
                "row_count": update.get("row_count", len(rows)),
                "truncated": bool(update.get("truncated")) or len(rows) > self.stream_rows,
            })]  # fmt: skip
        if node in FINAL_NODES:
            self.final = {
                "answer": update.get("answer"),
                "sql": self.sql if node != "decline" else None,
                "chart": update.get("chart"),
                "outcome": update.get("outcome"),
                "query_id": self.query_id,
            }
            return [("answer", self.final)]
        return []


def _interrupt_question(intr: Any) -> str | None:
    value = getattr(intr, "value", intr)
    return value.get("question") if isinstance(value, dict) else str(value)


async def run_turn(
    graph, inputs, config: dict, context, mapper: EventMapper
) -> AsyncIterator[Event]:
    """Yield SSE events for one turn. Errors become an `error` event; `done` is always last.

    (`done` is not yielded from a `finally`: if the client disconnects, the generator is closed
    and yielding there would raise. Nobody is listening for `done` in that case anyway.)
    """
    try:
        async for chunk in graph.astream(inputs, config, context=context, stream_mode="updates"):
            for event in mapper.feed(chunk):
                yield event
    except Exception as e:  # LLM provider down, DB down, bug
        log.exception("graph run failed")
        yield ("error", {"message": f"{type(e).__name__}: {e}"})
    yield ("done", {})
