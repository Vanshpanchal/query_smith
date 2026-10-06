"""Phase 2 graph: classify → (decline | select_tables → generate → validate → execute → check
→ answer), with every check able to loop back to generate_sql (repair) or stop (give_up).

Phase 3 adds contextualize (before classify) and the clarify interrupt.
"""

from langgraph.graph import END, START, StateGraph

from ask_data.graph.nodes import (
    AgentDeps,
    answer,
    check_result,
    classify,
    decline,
    execute_sql,
    generate_sql,
    give_up,
    select_tables,
    validate_sql,
)
from ask_data.graph.routing import route_after_check, route_intent
from ask_data.graph.state import AskState


def _check(ok: str) -> dict[str, str]:
    return {"ok": ok, "retry": "generate_sql", "give_up": "give_up"}


def build_graph(checkpointer=None):
    g = StateGraph(AskState, context_schema=AgentDeps)
    for fn in (classify, decline, select_tables, generate_sql, validate_sql, execute_sql,
               check_result, answer, give_up):  # fmt: skip
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", route_intent, {"sql": "select_tables", "decline": "decline"}
    )
    g.add_edge("select_tables", "generate_sql")
    g.add_conditional_edges("generate_sql", route_after_check, _check("validate_sql"))
    g.add_conditional_edges("validate_sql", route_after_check, _check("execute_sql"))
    g.add_conditional_edges("execute_sql", route_after_check, _check("check_result"))
    g.add_conditional_edges("check_result", route_after_check, _check("answer"))
    for node in ("answer", "decline", "give_up"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)
