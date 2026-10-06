from decimal import Decimal

from langgraph.types import Interrupt

from ask_data.api.streaming import EventMapper
from ask_data.config import settings
from ask_data.graph.routing import route_after_check, route_intent
from ask_data.graph.state import per_turn_reset
from ask_data.sql.executor import json_safe


def test_route_intent():
    assert route_intent({"intent": "decline"}) == "decline"
    assert route_intent({}) == "sql"


def test_route_after_check():
    assert route_after_check({"error": None, "attempts": 1}) == "ok"
    assert route_after_check({"error": "boom", "attempts": 1}) == "retry"
    assert route_after_check({"error": "boom", "attempts": settings.max_attempts}) == "give_up"


def test_per_turn_reset_clears_loop_fields():
    inputs = per_turn_reset("shop", "how many?")
    assert inputs["attempts"] == 0 and inputs["error"] is None and inputs["sql"] is None
    assert inputs["empty_retried"] is False and inputs["rows"] == []
    assert "last_sql" not in inputs  # persists across turns for follow-ups


def test_event_mapping_retry_flow():
    m = EventMapper("qid", stream_rows=2)
    events = []
    for chunk in [
        {"classify": {"intent": "sql"}},
        {"select_tables": {"tables": ["a", "b"], "schema_context": "..."}},
        {"generate_sql": {"sql": "SELECT x FROM a", "attempts": 1, "reasoning": "r1"}},
        {"validate_sql": {"error": "Unknown table", "error_stage": "validate"}},
        {"generate_sql": {"sql": "SELECT y FROM a", "attempts": 2, "reasoning": "r2"}},
        {"validate_sql": {"sql": "SELECT y FROM a LIMIT 1000", "error": None}},
        {"execute_sql": {"columns": ["y"], "rows": [[1], [2], [3]], "row_count": 3}},
        {"check_result": {"error": None}},
        {"answer": {"answer": "3 rows", "outcome": "answered", "chart": None}},
    ]:
        events += m.feed(chunk)

    names = [e[0] for e in events]
    assert names == ["status", "tables", "sql", "retry", "sql", "status", "rows", "answer"]
    assert events[3][1] == {"stage": "validate", "error": "Unknown table", "attempt": 1}
    assert events[4][1]["attempt"] == 2
    rows = events[6][1]
    assert rows["rows"] == [[1], [2]] and rows["row_count"] == 3 and rows["truncated"] is True
    final = events[7][1]
    assert final["sql"] == "SELECT y FROM a LIMIT 1000" and final["query_id"] == "qid"
    assert final["outcome"] == "answered"


def test_event_mapping_generate_failure_is_retry():
    m = EventMapper("q")
    events = m.feed(
        {"generate_sql": {"attempts": 1, "error": "bad json", "error_stage": "generate"}}
    )
    assert events == [("retry", {"stage": "generate", "error": "bad json", "attempt": 1})]


def test_event_mapping_decline_has_no_sql():
    m = EventMapper("q")
    (event,) = m.feed({"decline": {"answer": "no", "outcome": "declined"}})
    assert event[0] == "answer" and event[1]["sql"] is None and event[1]["outcome"] == "declined"


def test_event_mapping_interrupt():
    m = EventMapper("q")
    events = m.feed({"__interrupt__": (Interrupt(value={"question": "Runs or wickets?"}),)})
    assert events == [("clarify", {"question": "Runs or wickets?"})]


def test_json_safe():
    from datetime import date

    assert json_safe(Decimal("12.00")) == 12
    assert json_safe(Decimal("7.25")) == 7.25
    assert json_safe(date(2024, 1, 2)) == "2024-01-02"
    assert json_safe((1, Decimal("1.5"))) == [1, 1.5]
