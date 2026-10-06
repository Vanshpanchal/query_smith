from typing import Literal

from ask_data.config import settings


def route_intent(state) -> Literal["sql", "clarify", "decline"]:
    return state.get("intent") or "sql"


def route_after_check(state) -> Literal["ok", "retry", "give_up"]:
    """After generate/validate/execute/check: continue, loop back to generate_sql, or stop.

    `attempts` counts generate_sql calls, so MAX_ATTEMPTS=3 means at most 3 SQL drafts.
    """
    if not state.get("error"):
        return "ok"
    return "retry" if state.get("attempts", 0) < settings.max_attempts else "give_up"
