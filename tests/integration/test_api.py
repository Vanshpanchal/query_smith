"""API end-to-end over HTTP with the real lifespan: Postgres (both roles), checkpointer, registry,
Redis and the real executor. Only the LLMs are scripted, so the SQL really runs on the reader role.

Phase 2 DoD: the stream shows sql → retry → sql → rows → answer on BOTH ipl and sales threads.
"""

import json
import uuid

import httpx
import pytest
from conftest import ANSWER, SQL_INTENT, ScriptedLLM, clarify_intent, draft, requires_db, rewrite
from psycopg.rows import tuple_row

from ask_data.api.cache import AnswerCache, cache_key
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
        app.state.answer_cache = AnswerCache(app.state.redis, ttl_s=0)  # tests opt in
        created: list[str] = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.created_threads = created
            yield c
        async with app.state.app_pool.connection() as conn:  # clean up test threads
            for tid in created:
                await app.state.graph.checkpointer.adelete_thread(tid)
                await conn.execute(
                    "DELETE FROM app.feedback WHERE query_id IN "
                    "(SELECT id FROM app.query_log WHERE thread_id = %s)",
                    (tid,),
                )
                await conn.execute("DELETE FROM app.query_log WHERE thread_id = %s", (tid,))
                await conn.execute("DELETE FROM app.threads WHERE thread_id = %s", (tid,))


def script_llms(sql_drafts, intents=(SQL_INTENT,), rewrites=("(rewritten)",)) -> None:
    deps = app.state.deps
    deps.intent_llm = ScriptedLLM(*intents)
    deps.sql_llm = ScriptedLLM(*[draft(s) for s in sql_drafts])
    deps.answer_llm = ScriptedLLM(ANSWER)
    deps.rewrite_llm = ScriptedLLM(*[rewrite(q) for q in rewrites])


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


# --- Phase 3: clarify → resume, follow-ups, thread view, query log, feedback, retrieval -------


def any_dataset():
    ds = app.state.deps.registry.get(min(d.name for d in app.state.deps.registry.all()))
    return ds, f"SELECT COUNT(*) AS n FROM {ds.schema}.{min(ds.tables)}"


async def log_rows(thread_id: str) -> list[tuple]:
    async with app.state.app_pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            "SELECT outcome, attempts, final_sql, row_count, model FROM app.query_log "
            "WHERE thread_id = %s ORDER BY created_at",
            (thread_id,),
        )
        return await cur.fetchall()


async def test_clarify_then_resume_over_http(client):
    ds, count_sql = any_dataset()
    tid = await new_thread(client, ds.name)
    script_llms([count_sql], intents=(clarify_intent("Do you mean rows or tables?"), SQL_INTENT),
                rewrites=("How many rows are there?",))  # fmt: skip

    r = await client.post(f"/api/v1/threads/{tid}/ask", json={"question": "How big?"}, headers=AUTH)
    names = [n for n, _ in parse_sse(r.text)]
    assert names[-2:] == ["clarify", "done"] and "answer" not in names
    assert dict(parse_sse(r.text))["clarify"] == {"question": "Do you mean rows or tables?"}

    view = (await client.get(f"/api/v1/threads/{tid}", headers=AUTH)).json()
    assert view["pending_clarification"] == "Do you mean rows or tables?"
    assert view["last_query"]["outcome"] == "clarified"

    r = await client.post(f"/api/v1/threads/{tid}/resume", json={"reply": "rows"}, headers=AUTH)
    events = parse_sse(r.text)
    flow = [n for n, _ in events if n not in {"status", "tables"}]
    assert flow == ["sql", "rows", "answer", "done"], events
    status = [d for n, d in events if n == "status" and d["node"] == "contextualize"]
    assert status[0]["question"] == "How many rows are there?"

    view = (await client.get(f"/api/v1/threads/{tid}", headers=AUTH)).json()
    assert view["pending_clarification"] is None
    assert [m["content"] for m in view["messages"]][:3] == [
        "How big?",
        "Do you mean rows or tables?",
        "rows",
    ]
    assert [row[0] for row in await log_rows(tid)] == ["clarified", "answered"]


async def test_resume_errors(client):
    ds, _ = any_dataset()
    tid = await new_thread(client, ds.name)
    r = await client.post(f"/api/v1/threads/{tid}/resume", json={"reply": "x"}, headers=AUTH)
    assert r.status_code == 409
    r = await client.post("/api/v1/threads/nope/resume", json={"reply": "x"}, headers=AUTH)
    assert r.status_code == 404
    assert (await client.get("/api/v1/threads/nope", headers=AUTH)).status_code == 404


async def test_follow_up_and_thread_view(client):
    ds, count_sql = any_dataset()
    tid = await new_thread(client, ds.name)
    script_llms([count_sql, count_sql + " WHERE false"], rewrites=("Count with no rows?",))
    for q in ("How many rows?", "now with no rows"):
        r = await client.post(f"/api/v1/threads/{tid}/ask", json={"question": q}, headers=AUTH)
        assert parse_sse(r.text)[-1][0] == "done"

    sql_prompt = app.state.deps.sql_llm.prompts[1]
    assert "Question: Count with no rows?" in sql_prompt and count_sql in sql_prompt
    view = (await client.get(f"/api/v1/threads/{tid}", headers=AUTH)).json()
    assert [m["role"] for m in view["messages"]] == ["user", "assistant"] * 2
    assert view["last_sql"].endswith("WHERE FALSE LIMIT 1000")  # normalized by the guard
    rows = await log_rows(tid)
    assert [r[0] for r in rows] == ["answered", "answered"]
    assert rows[0][2].startswith("SELECT COUNT(*)") and rows[0][3] == 1
    assert rows[0][4] == settings.llm_model


async def test_feedback(client):
    ds, count_sql = any_dataset()
    script_llms([count_sql])
    body = (
        await client.post("/api/v1/ask", json={"question": "How many?", "dataset": ds.name},
                          headers=AUTH)
    ).json()  # fmt: skip
    client.created_threads.append(body["thread_id"])
    qid = body["query_id"]

    url = f"/api/v1/queries/{qid}/feedback"
    assert (await client.post(url, json={"rating": 1}, headers=AUTH)).status_code == 200
    r = await client.post(url, json={"rating": -1, "comment": "wrong"}, headers=AUTH)
    assert r.json() == {"query_id": qid, "rating": -1}
    async with app.state.app_pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute("SELECT rating, comment FROM app.feedback WHERE query_id = %s", (qid,))
        assert await cur.fetchall() == [(-1, "wrong")]  # replaced, not duplicated

    unknown = "/api/v1/queries/00000000-0000-0000-0000-000000000000/feedback"
    assert (await client.post(unknown, json={"rating": 1}, headers=AUTH)).status_code == 404
    bad = await client.post(url, json={"rating": 5}, headers=AUTH)
    assert bad.status_code == 422
    assert (await client.post("/api/v1/queries/x/feedback", json={"rating": 1},
                              headers=AUTH)).status_code == 422  # fmt: skip


async def test_one_shot_ask_reports_clarification(client):
    ds, count_sql = any_dataset()
    script_llms([count_sql], intents=(clarify_intent("Do you mean rows or tables?"),))
    r = await client.post("/api/v1/ask", json={"question": "How big?", "dataset": ds.name},
                          headers=AUTH)  # fmt: skip
    body = r.json()
    client.created_threads.append(body["thread_id"])
    assert (
        body["outcome"] == "clarified"
        and body["clarifying_question"] == "Do you mean rows or tables?"
    )
    assert body["answer"] is None and body["sql"] is None


async def test_pgvector_retrieval_on_real_cards(client):
    """Cards of a real dataset, stored under a throwaway name, searched with real pgvector."""
    if not loaded("ipl"):
        pytest.skip("dataset 'ipl' not loaded")
    from ask_data.schema.cards import build_cards
    from ask_data.schema.retrieval import (
        HashingEmbedder,
        PgVectorRetriever,
        expand_with_fks,
        upsert_cards,
    )  # fmt: skip

    ds, name, emb = app.state.deps.registry.get("ipl"), "pytest_cards", HashingEmbedder()
    async with app.state.app_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO app.datasets (name, schema_name, title, version, table_count, row_count)"
            " VALUES (%s, %s, 't', 'v', 0, 0) ON CONFLICT (name) DO NOTHING",
            (name, name),
        )
        try:
            cards = await build_cards(conn, ds.schema_info)
            await upsert_cards(conn, name, cards, await emb.embed_documents(list(cards.values())),
                               emb.model)  # fmt: skip
            hits = await PgVectorRetriever(app.state.app_pool, emb).search(
                name, "Which bowler took the most wickets, by wicket kind?", 2
            )
        finally:
            await conn.execute("DELETE FROM app.datasets WHERE name = %s", (name,))  # cascades
    assert hits[0] == "deliveries", hits
    assert {"deliveries", "players"} <= set(expand_with_fks(hits, ds.schema_info))


async def test_answer_cache_serves_a_repeated_first_question(client):
    name = min(d.name for d in app.state.deps.registry.all())
    ds = app.state.deps.registry.get(name)
    table = min(ds.tables)
    app.state.answer_cache = AnswerCache(app.state.redis, ttl_s=60)
    question = f"Cache test {uuid.uuid4()}: how many rows?"
    key = cache_key(name, ds.version, question, settings.llm_model)
    try:
        script_llms([f"SELECT COUNT(*) AS n FROM {ds.schema}.{table}"])
        first = (await client.post("/api/v1/ask", headers=AUTH,
                                   json={"question": question, "dataset": name})).json()  # fmt: skip
        client.created_threads.append(first["thread_id"])
        assert first["outcome"] == "answered" and first["cached"] is False

        # Same question, other casing/punctuation, new thread: no LLM call, same answer
        script_llms(["SELECT nope FROM nowhere"])
        tid = await new_thread(client, name)
        r = await client.post(f"/api/v1/threads/{tid}/ask", headers=AUTH,
                              json={"question": f"  {question.upper()} "})  # fmt: skip
        events = parse_sse(r.text)
        flow = [n for n, _ in events]
        assert flow == ["status", "tables", "sql", "rows", "answer", "done"], events
        answer = next(d for n, d in events if n == "answer")
        assert answer["cached"] is True and answer["sql"] == first["sql"]
        assert app.state.deps.sql_llm.prompts == []

        # The cached turn is in the thread (follow-ups work) and logged as cached
        state = await app.state.graph.aget_state({"configurable": {"thread_id": tid}})
        assert state.values["last_sql"] == first["sql"] and len(state.values["messages"]) == 2
        async with (
            app.state.app_pool.connection() as conn,
            conn.cursor(row_factory=tuple_row) as cur,
        ):
            await cur.execute("SELECT outcome, cached FROM app.query_log WHERE id = %s",
                              (answer["query_id"],))  # fmt: skip
            assert await cur.fetchone() == ("answered", True)

        # A follow-up on that thread is never served from the cache
        script_llms([f"SELECT COUNT(*) AS n FROM {ds.schema}.{table}"], rewrites=[question])
        again = (await client.post("/api/v1/ask", headers=AUTH,
                                   json={"question": question, "thread_id": tid})).json()  # fmt: skip
        assert again["cached"] is False and app.state.deps.sql_llm.prompts
    finally:
        await app.state.redis.delete(key)


async def test_answer_cache_skips_failed_answers(client):
    name = min(d.name for d in app.state.deps.registry.all())
    ds = app.state.deps.registry.get(name)
    app.state.answer_cache = AnswerCache(app.state.redis, ttl_s=60)
    question = f"Cache test {uuid.uuid4()}: broken?"
    key = cache_key(name, ds.version, question, settings.llm_model)
    script_llms(["SELECT nope FROM nowhere"])
    body = (await client.post("/api/v1/ask", headers=AUTH,
                              json={"question": question, "dataset": name})).json()  # fmt: skip
    client.created_threads.append(body["thread_id"])
    assert body["outcome"] == "gave_up"
    assert await app.state.redis.get(key) is None


async def test_demo_page_and_web_config(client, monkeypatch):
    r = await client.get("/")
    assert r.status_code == 200 and "<title>AskData</title>" in r.text
    monkeypatch.setattr(settings, "demo_api_key", "not-a-real-key")
    assert (await client.get("/web-config")).json() == {"demo_api_key": None}  # not in API_KEYS
    monkeypatch.setattr(settings, "demo_api_key", API_KEY)
    assert (await client.get("/web-config")).json() == {"demo_api_key": API_KEY}
