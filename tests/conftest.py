"""Shared fakes: scripted LLMs, an in-memory executor and a tiny dataset that needs no database.

The fake dataset is deliberately NOT one of the real ones, so graph tests prove the agent works
on whatever the registry hands it.
"""

from collections.abc import Callable
from typing import Any

import psycopg
import pytest

from ask_data.config import settings
from ask_data.datasets.config import DatasetCfg
from ask_data.datasets.registry import Dataset, DatasetRegistry, Fewshot
from ask_data.graph.models import FinalAnswer, Intent, SqlDraft, StandaloneQuestion
from ask_data.graph.nodes import AgentDeps
from ask_data.schema.catalog import ColumnInfo, SchemaInfo, TableInfo
from ask_data.sql.executor import QueryError, QueryResult


class ScriptedLLM:
    """Returns the queued outputs in order (an Exception instance is raised instead).
    The last output repeats once the queue is empty. Every prompt is recorded."""

    def __init__(self, *outputs: Any):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str, /) -> Any:
        self.prompts.append(prompt)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


class FakeExecutor:
    """`results` maps a substring of the (guarded) SQL to rows, or to a QueryError to raise."""

    def __init__(self, results: dict[str, Any] | None = None, cost: float = 10.0):
        self.results = results or {}
        self.cost = cost
        self.executed: list[str] = []

    async def explain_cost(self, query: str, schema: str) -> float:
        return self.cost

    async def run_select(self, query: str, schema: str, max_rows: int) -> QueryResult:
        self.executed.append(query)
        for needle, result in self.results.items():
            if needle in query:
                if isinstance(result, Exception):
                    raise result
                columns, rows = result
                return QueryResult(
                    columns, [list(r) for r in rows[:max_rows]], len(rows) > max_rows
                )
        raise QueryError(f"UndefinedColumn: no fake result for {query!r}")


def make_dataset(name: str = "shop") -> Dataset:
    cfg = DatasetCfg.model_validate(
        {
            "name": name,
            "title": f"{name.title()} test data",
            "description": "Items and orders of a toy shop.",
            "source": {"type": "csv"},
            "tables": {"items": {}, "orders": {}},
        }
    )
    schema_info = SchemaInfo(
        name,
        {
            "items": TableInfo(
                "items",
                "One row per item.",
                [ColumnInfo("id", "bigint"), ColumnInfo("name", "text", "Item name")],
                ["PRIMARY KEY (id)"],
            ),
            "orders": TableInfo(
                "orders",
                None,
                [ColumnInfo("id", "bigint"), ColumnInfo("item_id", "bigint")],
                ["PRIMARY KEY (id)", "FOREIGN KEY (item_id) REFERENCES items(id)"],
            ),
        },
    )
    return Dataset(
        cfg=cfg,
        schema_info=schema_info,
        version="v1",
        notes="- Count orders with COUNT(*).",
        fewshots=[Fewshot("How many items are there?", f"SELECT COUNT(*) FROM {name}.items")],
    )


def draft(sql: str) -> SqlDraft:
    return SqlDraft(reasoning="test", sql=sql, tables_used=[])


SQL_INTENT = Intent(intent="sql", reason="about the data")
DECLINE_INTENT = Intent(intent="decline", reason="off-topic")


def clarify_intent(question: str = "By count or by value?") -> Intent:
    return Intent(intent="clarify", reason="ambiguous", clarifying_question=question)


def rewrite(question: str) -> StandaloneQuestion:
    return StandaloneQuestion(question=question)


ANSWER = FinalAnswer(answer="The answer.")


@pytest.fixture
def make_deps() -> Callable[..., AgentDeps]:
    def _make(
        sql: list[Any] | None = None,
        intent: Any = SQL_INTENT,
        answer: Any = ANSWER,
        results: dict[str, Any] | None = None,
        cost: float = 10.0,
        datasets: list[Dataset] | None = None,
        rewrites: list[Any] | None = None,
        retriever: Any = None,
    ) -> AgentDeps:
        datasets = datasets or [make_dataset()]
        return AgentDeps(
            registry=DatasetRegistry({d.name: d for d in datasets}),
            executor=FakeExecutor(results, cost),
            intent_llm=intent if isinstance(intent, ScriptedLLM) else ScriptedLLM(intent),
            sql_llm=ScriptedLLM(*(sql or [draft("SELECT name FROM items")])),
            answer_llm=ScriptedLLM(answer),
            rewrite_llm=ScriptedLLM(*(rewrites or [rewrite("(rewritten)")])),
            retriever=retriever,
        )

    return _make


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.reader_database_url, connect_timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


DB_AVAILABLE = _db_reachable()
requires_db = pytest.mark.skipif(
    not DB_AVAILABLE, reason="Postgres (docker compose up) not reachable"
)
