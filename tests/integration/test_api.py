"""API end-to-end over HTTP with the real lifespan: Postgres (both roles), checkpointer, registry,
Redis and the real executor. Only the LLMs are scripted, so the SQL really runs on the reader role.

Phase 2 DoD: the stream shows sql → retry → sql → rows → answer on BOTH ipl and sales threads.
"""

import json

import httpx
import pytest
from conftest import ANSWER, SQL_INTENT, ScriptedLLM, draft, requires_db
from psycopg.rows import tuple_row

from ask_data.api.security import RateLimiter
from ask_data.config import settings
from ask_data.main import app

pytestmark = requires_db

API_KEY = min(settings.api_key_set)
AUTH = {"X-API-Key": API_KEY}
# A first draft with a wrong column (caught by EXPLAIN in validate_sql), then a correct one
RETRY_SQL = {
    "ipl": ("SELECT nme FROM ipl.teams", "SELECT name FROM ipl.teams ORDER BY id LIMIT 3"),
    "sales": (
        "SELECT nme FROM sales.products",
        "SELECT name FROM sales.products ORDER BY product_id LIMIT 3",
    ),
}


@pytest.fixture
async def client():
    async with app.router.lifespan_context(app):
        app.state.rate_limiter = RateLimiter(app.state.redis, limit=10_000)
        created: list[str] = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.created_threads = created
            yield c
        async with app.state.app_pool.connection() as conn:  # clean up test threads
            for tid in created:
                await app.state.graph.checkpointer.adelete_thread(tid)
                await conn.execute("DELETE FROM app.threads WHERE thread_id = %s", (tid,))


def script_llms(sql_drafts) -> None:
    deps = app.state.deps
    deps.intent_llm = ScriptedLLM(SQL_INTENT)
    deps.sql_llm = ScriptedLLM(*[draft(s) for s in sql_drafts])
    deps.answer_llm = ScriptedLLM(ANSWER)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events


async def new_thread(client, dataset: str) -> str:
    r = await client.post("/api/v1/threads", json={"dataset": dataset}, headers=AUTH)
    assert r.status_code == 200, r.text
    tid = r.json()["thread_id"]
    client.created_threads.append(tid)
    return tid


def loaded(name: str) -> bool:
    return name in app.state.deps.registry


async def test_health_and_ready(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    r = await client.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json()["checks"] == {"postgres_app": "ok", "postgres_reader": "ok", "redis": "ok"}


async def test_auth_required(client):
    assert (await client.get("/api/v1/datasets")).status_code == 401
    bad = await client.get("/api/v1/datasets", headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401


async def test_datasets_and_schema(client):
    r = await client.get("/api/v1/datasets", headers=AUTH)
    names = {d["name"] for d in r.json()}
    assert names and all(d["tables"] for d in r.json())
    name = min(names)
    schema = (await client.get(f"/api/v1/datasets/{name}/schema", headers=AUTH)).json()
    assert schema["tables"] and schema["tables"][0]["columns"]
    missing = await client.get("/api/v1/datasets/nope/schema", headers=AUTH)
    assert missing.status_code == 404


async def test_unknown_dataset_is_404(client):
    r = await client.post("/api/v1/threads", json={"dataset": "nope"}, headers=AUTH)
    assert r.status_code == 404


async def test_question_validation(client):
    r = await client.post("/api/v1/threads/x/ask", json={"question": "a"}, headers=AUTH)
    assert r.status_code == 422


@pytest.mark.parametrize("dataset", ["ipl", "sales"])
async def test_sse_retry_flow(client, dataset):
    if not loaded(dataset):
        pytest.skip(f"dataset {dataset!r} not loaded (make load DATASET={dataset})")
    tid = await new_thread(client, dataset)
    script_llms(RETRY_SQL[dataset])

    r = await client.post(
        f"/api/v1/threads/{tid}/ask", json={"question": "Name three of them"}, headers=AUTH
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    flow = [name for name, _ in events if name not in {"status", "tables"}]
    assert flow == ["sql", "retry", "sql", "rows", "answer", "done"], events

    data = dict(reversed(events))  # first occurrence of each event name
    assert data["retry"]["stage"] == "validate" and "nme" in data["retry"]["error"]
    rows = next(d for n, d in events if n == "rows")
    assert rows["row_count"] == 3 and len(rows["rows"]) == 3
    answer = next(d for n, d in events if n == "answer")
    assert answer["outcome"] == "answered" and answer["query_id"]
    assert answer["sql"].endswith("LIMIT 3")


async def test_one_shot_ask_and_follow_up_state(client):
    name = min(d.name for d in app.state.deps.registry.all())
    ds = app.state.deps.registry.get(name)
    table = min(ds.tables)
    script_llms([f"SELECT COUNT(*) AS n FROM {ds.schema}.{table}"])

    r = await client.post(
        "/api/v1/ask", json={"question": "How many rows?", "dataset": name}, headers=AUTH
    )
    assert r.status_code == 200, r.text
    body = r.json()
    client.created_threads.append(body["thread_id"])
    assert body["outcome"] == "answered" and body["attempts"] == 1
    assert body["columns"] == ["n"] and body["rows"][0][0] > 0

    # same thread again: the per-turn reset starts the loop fresh, history is kept
    script_llms(["SELECT nope FROM nowhere"])
    again = await client.post(
        "/api/v1/ask", headers=AUTH, json={"question": "And again?", "thread_id": body["thread_id"]}
    )
    assert again.json()["outcome"] == "gave_up"
    assert again.json()["attempts"] == settings.max_attempts
    state = await app.state.graph.aget_state({"configurable": {"thread_id": body["thread_id"]}})
    assert len(state.values["messages"]) == 4
    assert state.values["last_sql"].startswith("SELECT COUNT(*)")  # from the answered turn


async def test_thread_bound_to_dataset_in_db(client):
    name = min(d.name for d in app.state.deps.registry.all())
    tid = await new_thread(client, name)
    async with app.state.app_pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute("SELECT dataset FROM app.threads WHERE thread_id = %s", (tid,))
        assert (await cur.fetchone())[0] == name


async def test_llm_down_gives_error_free_give_up(client):
    name = min(d.name for d in app.state.deps.registry.all())
    tid = await new_thread(client, name)
    script_llms(["x"])
    app.state.deps.sql_llm = ScriptedLLM(RuntimeError("provider down"))
    r = await client.post(
        f"/api/v1/threads/{tid}/ask", json={"question": "anything?"}, headers=AUTH
    )
    answer = next(d for n, d in parse_sse(r.text) if n == "answer")
    assert answer["outcome"] == "gave_up" and "provider down" in answer["answer"]


async def test_rate_limit(client):
    app.state.rate_limiter = RateLimiter(app.state.redis, limit=0)
    r = await client.get("/api/v1/datasets", headers=AUTH)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


async def test_admin_reload_requires_admin_key(client):
    assert (await client.post("/admin/datasets/reload", headers=AUTH)).status_code == 401
    if settings.admin_api_key:
        r = await client.post(
            "/admin/datasets/reload", headers={"X-API-Key": settings.admin_api_key}
        )
        assert r.status_code == 200 and r.json()["datasets"]
