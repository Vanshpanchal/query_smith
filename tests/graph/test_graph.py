"""Paths through the compiled graph with scripted LLMs and a fake executor (no DB, no API key)."""

import pytest
from conftest import DECLINE_INTENT, SQL_INTENT, ScriptedLLM, clarify_intent, draft, rewrite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ask_data.config import settings
from ask_data.graph.builder import build_graph
from ask_data.graph.models import FinalAnswer, Intent
from ask_data.graph.state import per_turn_reset
from ask_data.sql.executor import QueryError

ITEMS = (["name"], [("Robot",), ("Kite",)])


@pytest.fixture
def graph():
    return build_graph(InMemorySaver())


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


async def ask(graph, deps, question: str, thread: str = "t1") -> dict:
    return await graph.ainvoke(per_turn_reset("shop", question), cfg(thread), context=deps)


async def test_happy_path(graph, make_deps):
    deps = make_deps(results={"FROM items": ITEMS})
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and out["attempts"] == 1
    assert out["sql"] == "SELECT name FROM items LIMIT 1000"  # guard added the limit
    assert out["rows"] == [["Robot"], ["Kite"]] and out["last_sql"] == out["sql"]
    assert out["messages"][-1].content == "The answer."
    sql_prompt = deps.sql_llm.prompts[0]
    assert "TABLE shop.items" in sql_prompt and "Count orders with COUNT(*)" in sql_prompt
    assert "previous query failed" not in sql_prompt


async def test_guard_error_then_success(graph, make_deps):
    deps = make_deps(
        sql=[draft("SELECT name FROM itemz"), draft("SELECT name FROM items")],
        results={"FROM items": ITEMS},
    )
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and out["attempts"] == 2
    repair = deps.sql_llm.prompts[1]
    assert "SELECT name FROM itemz" in repair and "Unknown table: shop.itemz" in repair


async def test_db_error_then_success(graph, make_deps):
    deps = make_deps(
        sql=[draft("SELECT nme FROM items"), draft("SELECT name FROM items")],
        results={
            "nme": QueryError('UndefinedColumn: column "nme" does not exist'),
            "FROM items": ITEMS,
        },
    )
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and out["attempts"] == 2
    assert 'column "nme" does not exist' in deps.sql_llm.prompts[1]


async def test_gives_up_after_max_attempts(graph, make_deps):
    deps = make_deps(sql=[draft("SELECT name FROM nope")])
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "gave_up"
    assert out["attempts"] == settings.max_attempts
    assert len(deps.sql_llm.prompts) == settings.max_attempts
    assert "Unknown table" in out["answer"] and deps.executor.executed == []


async def test_plan_cost_cap_triggers_retry(graph, make_deps):
    deps = make_deps(cost=settings.max_plan_cost * 10)
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "gave_up" and "plan cost" in out["error"]
    assert deps.executor.executed == []  # EXPLAIN blocked it before execution


async def test_structured_output_failure_counts_as_attempt(graph, make_deps):
    deps = make_deps(
        sql=[ValueError("bad json"), ValueError("bad json"), draft("SELECT name FROM items")],
        results={"FROM items": ITEMS},
    )
    out = await ask(graph, deps, "list items")
    # attempt 1: both tries (call + one retry) fail → retry; attempt 2 succeeds
    assert out["outcome"] == "answered" and out["attempts"] == 2


async def test_empty_result_retried_once(graph, make_deps):
    deps = make_deps(
        sql=[
            draft("SELECT name FROM items WHERE name = 'robot'"),
            draft("SELECT name FROM items WHERE name ILIKE 'robot'"),
        ],
        results={"= 'robot'": (["name"], []), "ILIKE": (["name"], [("Robot",)])},
    )
    out = await ask(graph, deps, "is there a robot?")
    assert out["attempts"] == 2 and out["rows"] == [["Robot"]]
    assert "returned no rows" in deps.sql_llm.prompts[1]


async def test_empty_result_accepted_after_hint(graph, make_deps):
    deps = make_deps(results={"FROM items": (["name"], [])})
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and out["attempts"] == 2 and out["rows"] == []


async def test_decline(graph, make_deps):
    deps = make_deps(intent=DECLINE_INTENT)
    out = await ask(graph, deps, "what's the weather?")
    assert out["outcome"] == "declined" and out["attempts"] == 0
    assert "How many items are there?" in out["answer"]  # example questions from fewshots
    assert deps.sql_llm.prompts == []


async def test_classify_failure_fails_open_to_sql(graph, make_deps):
    deps = make_deps(intent=RuntimeError("provider down"), results={"FROM items": ITEMS})
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered"


async def test_answer_llm_failure_still_returns_rows(graph, make_deps):
    deps = make_deps(answer=RuntimeError("provider down"), results={"FROM items": ITEMS})
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and "2 row(s)" in out["answer"]


async def test_second_question_resets_attempts(graph, make_deps):
    # Regression for Architecture gotcha #1: state persists per thread, the loop fields must not
    failing = make_deps(sql=[draft("SELECT x FROM nope")])
    first = await ask(graph, failing, "q1", thread="same")
    assert first["outcome"] == "gave_up"

    ok = make_deps(results={"FROM items": ITEMS})
    second = await ask(graph, ok, "q2", thread="same")
    assert second["outcome"] == "answered" and second["attempts"] == 1
    assert len(second["messages"]) == 4  # q1, give-up, q2, answer: history persists


async def test_answer_sees_at_most_answer_rows(graph, make_deps):
    many = (["n"], [(i,) for i in range(100)])
    deps = make_deps(results={"FROM items": many})
    await ask(graph, deps, "list items")
    prompt = deps.answer_llm.prompts[0]
    assert f"({settings.answer_rows - 1},)" in prompt and f"({settings.answer_rows},)" not in prompt


async def test_dataset_isolation(graph, make_deps):
    # A thread on dataset "shop" can't read dataset "zoo", even if the model tries
    from conftest import make_dataset

    deps = make_deps(
        sql=[draft("SELECT name FROM zoo.items")],
        datasets=[make_dataset("shop"), make_dataset("zoo")],
    )
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "gave_up" and "'zoo' is not allowed" in out["error"]


def test_scripted_llm_repeats_last():
    llm = ScriptedLLM(1, 2)
    import asyncio

    assert [asyncio.run(llm.ainvoke("p")) for _ in range(3)] == [1, 2, 2]


# --- Phase 3: follow-ups, clarification, schema retrieval -----------------------------------


async def test_first_question_skips_contextualize_llm(graph, make_deps):
    deps = make_deps(results={"FROM items": ITEMS})
    out = await ask(graph, deps, "list items")
    assert out["question"] == "list items" and deps.rewrite_llm.prompts == []


async def test_follow_up_is_rewritten_with_history_and_last_sql(graph, make_deps):
    deps = make_deps(
        sql=[draft("SELECT name FROM items"), draft("SELECT name FROM items WHERE id > 1")],
        results={"FROM items": ITEMS},
        rewrites=[rewrite("List items with id above 1")],
    )
    await ask(graph, deps, "list items", thread="f")
    out = await ask(graph, deps, "now only id above 1", thread="f")

    assert out["question"] == "List items with id above 1"
    history = deps.rewrite_llm.prompts[0]
    assert "User: list items" in history and "Assistant: The answer." in history
    assert "User: now only id above 1" in history
    assert "SELECT name FROM items LIMIT 1000" in history  # last_sql of turn 1
    sql_prompt = deps.sql_llm.prompts[1]
    assert "Question: List items with id above 1" in sql_prompt
    assert "SQL of the previous question" in sql_prompt
    assert out["last_sql"] == "SELECT name FROM items WHERE id > 1 LIMIT 1000"


async def test_contextualize_failure_falls_back_to_the_raw_question(graph, make_deps):
    deps = make_deps(results={"FROM items": ITEMS}, rewrites=[RuntimeError("provider down")])
    await ask(graph, deps, "list items", thread="f2")
    out = await ask(graph, deps, "and again", thread="f2")
    assert out["outcome"] == "answered" and out["question"] == "and again"


async def test_clarify_then_resume(graph, make_deps):
    deps = make_deps(
        intent=ScriptedLLM(clarify_intent("By count or by value?"), SQL_INTENT),
        results={"FROM items": ITEMS},
        rewrites=[rewrite("Which items sell best by count?")],
    )
    paused = await ask(graph, deps, "best items?", thread="c")
    assert paused["__interrupt__"][0].value == {"question": "By count or by value?"}
    snapshot = await graph.aget_state(cfg("c"))
    assert snapshot.interrupts and snapshot.next == ("clarify",)
    assert deps.sql_llm.prompts == []  # nothing ran past the pause

    out = await graph.ainvoke(Command(resume="by count"), cfg("c"), context=deps)
    assert out["outcome"] == "answered" and out["clarify_count"] == 1
    assert out["clarification"] == "by count"
    assert out["question"] == "Which items sell best by count?"
    contents = [m.content for m in out["messages"]]
    assert contents[:3] == ["best items?", "By count or by value?", "by count"]
    assert "User: by count" in deps.rewrite_llm.prompts[0]
    assert not (await graph.aget_state(cfg("c"))).interrupts


async def test_only_one_clarification_per_turn(graph, make_deps):
    # The model keeps asking; after one clarification classify must go to SQL (ADR-008)
    deps = make_deps(
        intent=ScriptedLLM(clarify_intent(), clarify_intent()),
        results={"FROM items": ITEMS},
    )
    await ask(graph, deps, "best items?", thread="c2")
    out = await graph.ainvoke(Command(resume="by value"), cfg("c2"), context=deps)
    assert out["outcome"] == "answered" and out["clarify_count"] == 1
    assert "clarify" + '" is NOT' in deps.intent_llm.prompts[1].replace("\n", " ")
    assert "clarify" + '" is NOT' not in deps.intent_llm.prompts[0].replace("\n", " ")


async def test_assumption_reaches_sql_and_answer(graph, make_deps):
    chosen = Intent(intent="sql", reason="ambiguous", assumption="best = most orders")
    deps = make_deps(intent=chosen, results={"FROM items": ITEMS})
    out = await ask(graph, deps, "best items?")
    assert out["assumption"] == "best = most orders"
    assert (
        "Interpretation to use (the question was ambiguous): best = most orders"
        in (deps.sql_llm.prompts[0])
    )
    assert "answered as: best = most orders" in deps.answer_llm.prompts[0]


async def test_new_question_on_a_paused_thread_starts_a_new_turn(graph, make_deps):
    deps = make_deps(
        intent=ScriptedLLM(clarify_intent(), SQL_INTENT), results={"FROM items": ITEMS}
    )
    await ask(graph, deps, "best items?", thread="c3")
    out = await ask(graph, deps, "list items", thread="c3")
    assert out["outcome"] == "answered" and out["clarify_count"] == 0
    assert out.get("clarifying_question") is None
    assert not (await graph.aget_state(cfg("c3"))).interrupts


class FakeRetriever:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def search(self, dataset: str, question: str, k: int) -> list[str]:
        self.calls.append((dataset, question, k))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_small_dataset_skips_retrieval(graph, make_deps):
    retriever = FakeRetriever(["orders"])
    deps = make_deps(results={"FROM items": ITEMS}, retriever=retriever)
    out = await ask(graph, deps, "list items")
    assert out["table_selection"] == "full" and retriever.calls == []


async def test_retrieval_with_fk_expansion(graph, make_deps, monkeypatch):
    monkeypatch.setattr(settings, "schema_retrieval_min_tables", 1)
    retriever = FakeRetriever(["orders"])
    deps = make_deps(results={"FROM items": ITEMS}, retriever=retriever)
    out = await ask(graph, deps, "list items")
    assert out["table_selection"] == "retrieved"
    assert out["tables"] == ["orders", "items"]  # orders.item_id → items joined in
    assert retriever.calls == [("shop", "list items", settings.schema_retrieval_top_k)]


@pytest.mark.parametrize("result", [[], RuntimeError("pgvector down")])
async def test_retrieval_problem_falls_back_to_full_schema(graph, make_deps, monkeypatch, result):
    monkeypatch.setattr(settings, "schema_retrieval_min_tables", 1)
    deps = make_deps(results={"FROM items": ITEMS}, retriever=FakeRetriever(result))
    out = await ask(graph, deps, "list items")
    assert out["outcome"] == "answered" and out["table_selection"] == "full"
    assert out["tables"] == ["items", "orders"]


async def test_unusable_clarification_is_answered_instead(graph, make_deps):
    label_only = Intent(intent="clarify", reason="ambiguous", clarifying_question="Best item")
    deps = make_deps(intent=label_only, results={"FROM items": ITEMS})
    out = await ask(graph, deps, "best item?")
    assert out["outcome"] == "answered" and not out.get("__interrupt__")


async def test_answer_returns_a_chart_spec(graph, make_deps):
    wins = (["name", "wins"], [("Robot", 3), ("Kite", 2)])
    answer = FinalAnswer(answer="Robot sold most.", chart="pie")
    deps = make_deps(results={"FROM items": wins}, answer=answer)
    out = await ask(graph, deps, "orders per item")
    assert out["chart"] == {"type": "pie", "x": "name", "y": "wins"}
    assert "Also pick `chart`" in deps.answer_llm.prompts[0]


async def test_chart_is_inferred_when_the_answer_llm_fails(graph, make_deps):
    wins = (["name", "wins"], [("Robot", 3), ("Kite", 2)])
    deps = make_deps(results={"FROM items": wins}, answer=RuntimeError("provider down"))
    out = await ask(graph, deps, "orders per item")
    assert out["chart"] == {"type": "bar", "x": "name", "y": "wins"}
