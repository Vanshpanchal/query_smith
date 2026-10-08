import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from ask_data.api import query_log, threads_repo
from ask_data.api.cache import cache_key
from ask_data.api.security import require_api_key
from ask_data.api.streaming import Event, EventMapper, run_turn
from ask_data.config import settings
from ask_data.graph.state import per_turn_reset

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["threads"])


class CreateThreadRequest(BaseModel):
    dataset: str | None = None  # None → settings.default_dataset


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class OneShotAskRequest(AskRequest):
    dataset: str | None = None
    thread_id: str | None = None  # continue an existing thread; default: a new one


class ResumeRequest(BaseModel):
    reply: str = Field(min_length=1, max_length=500)


def _resolve_dataset(request: Request, name: str | None) -> str:
    name = name or settings.default_dataset
    if not name:
        raise HTTPException(422, "No dataset given and DEFAULT_DATASET is not set")
    if name not in request.app.state.deps.registry:
        raise HTTPException(404, f"Dataset {name!r} is not loaded")
    return name


def _check_loaded(request: Request, dataset: str) -> str:
    if dataset not in request.app.state.deps.registry:
        raise HTTPException(404, f"Dataset {dataset!r} of this thread is no longer loaded")
    return dataset


async def _thread_dataset(request: Request, thread_id: str) -> str:
    pool = request.app.state.app_pool
    dataset = await threads_repo.get_dataset(pool, thread_id)
    if dataset is None:  # lazily bind an unknown thread id to the default dataset
        dataset = await threads_repo.get_or_create(pool, thread_id, _resolve_dataset(request, None))
    return _check_loaded(request, dataset)


async def _existing_thread_dataset(request: Request, thread_id: str) -> str:
    dataset = await threads_repo.get_dataset(request.app.state.app_pool, thread_id)
    if dataset is None:
        raise HTTPException(404, f"Unknown thread {thread_id!r}")
    return dataset


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class Turn:
    """One graph run (a new question, or a resume): streams events and logs the run in
    app.query_log before `done`, so a client that saw `done` can always send feedback.

    A thread's first question (`cacheable`) may be answered from the Redis answer cache: no graph
    run, but the answer is written into the thread's checkpoint so follow-ups still work."""

    def __init__(self, request: Request, thread_id: str, dataset: str, inputs, question: str,
                 cacheable: bool = False):  # fmt: skip
        self.request, self.thread_id, self.dataset = request, thread_id, dataset
        self.inputs, self.question = inputs, question
        self.mapper = EventMapper(query_id=str(uuid.uuid4()))
        self.started = time.perf_counter()
        self.state: dict = {}
        self.outcome: str | None = None
        self.cached = False
        self.cache_key: str | None = None
        if cacheable:
            version = request.app.state.deps.registry.get(dataset).version
            self.cache_key = cache_key(dataset, version, question, settings.llm_model)

    async def events(self) -> AsyncIterator[Event]:
        app = self.request.app
        config = _config(self.thread_id)
        if self.cache_key and not await self._first_question(config):
            self.cache_key = None  # a follow-up depends on the conversation: never cached
        if self.cache_key and (hit := await app.state.answer_cache.get(self.cache_key)):
            try:
                await self._store_in_thread(hit, config)
            except Exception:  # a cache problem only costs a normal run
                log.exception("could not replay a cached answer; running the graph")
            else:
                for event in self._cached_events(hit):
                    yield event
                await self._finish(config)
                yield "done", {}
                return
        async for name, data in run_turn(app.state.graph, self.inputs, config, app.state.deps,
                                         self.mapper):  # fmt: skip
            if name == "done":
                await self._finish(config)
            yield name, data

    async def _first_question(self, config: dict) -> bool:
        snapshot = await self.request.app.state.graph.aget_state(config)
        return not (snapshot.values or {}).get("messages")

    async def _store_in_thread(self, hit: dict, config: dict) -> None:
        values = {
            **per_turn_reset(self.dataset, self.question),
            **hit,
            "messages": [HumanMessage(self.question), AIMessage(hit["answer"])],
            "last_sql": hit["sql"],
            "outcome": "answered",
        }
        await self.request.app.state.graph.aupdate_state(config, values, as_node="answer")
        self.cached = True

    def _cached_events(self, hit: dict) -> list[Event]:
        """The events a real run would have sent, built by the same EventMapper."""
        chunks = [
            {"select_tables": {"tables": hit["tables"], "table_selection": hit["table_selection"]}},
            {"generate_sql": {"sql": hit["sql"], "attempts": hit["attempts"]}},
            {"execute_sql": {k: hit[k] for k in ("columns", "rows", "row_count", "truncated")}},
            {"answer": {"answer": hit["answer"], "chart": hit["chart"], "outcome": "answered"}},
        ]
        events: list[Event] = [("status", {"node": "cache", "message": "Answered from cache"})]
        for chunk in chunks:
            events.extend(self.mapper.feed(chunk))
        self.mapper.final["cached"] = True  # the `answer` event's payload
        return events

    async def run(self) -> None:
        async for _ in self.events():
            pass

    async def _finish(self, config: dict) -> None:
        try:
            snapshot = await self.request.app.state.graph.aget_state(config)
            self.state = snapshot.values or {}
            if self.mapper.error:
                self.outcome = "error"
            elif snapshot.interrupts:
                self.outcome = "clarified"
            else:
                self.outcome = self.state.get("outcome") or "error"
            await query_log.insert(
                self.request.app.state.app_pool,
                query_log.QueryLogRow(
                    id=self.mapper.query_id,
                    thread_id=self.thread_id,
                    dataset=self.dataset,
                    question=self.state.get("question") or self.question,
                    outcome=self.outcome,
                    final_sql=self.state.get("sql")
                    if self.outcome in {"answered", "gave_up"}
                    else None,
                    attempts=self.state.get("attempts", 0),
                    row_count=self.state.get("row_count") if self.outcome == "answered" else None,
                    latency_ms=round((time.perf_counter() - self.started) * 1000),
                    model=settings.llm_model,
                    cached=self.cached,
                ),
            )
            if self.cache_key and not self.cached and self.outcome == "answered":
                await self.request.app.state.answer_cache.put(self.cache_key, self.state)
        except Exception:  # logging must never break the answer
            log.exception("could not write app.query_log for thread %s", self.thread_id)

    def sse(self) -> EventSourceResponse:
        async def stream():
            async for name, data in self.events():
                yield {"event": name, "data": json.dumps(data, default=str)}

        return EventSourceResponse(stream())


async def _pending_question(request: Request, thread_id: str) -> str | None:
    snapshot = await request.app.state.graph.aget_state(_config(thread_id))
    for intr in snapshot.interrupts:
        value = getattr(intr, "value", intr)
        return value.get("question") if isinstance(value, dict) else str(value)
    return None


@router.post("/threads")
async def create_thread(body: CreateThreadRequest, request: Request) -> dict:
    dataset = _resolve_dataset(request, body.dataset)
    thread_id = str(uuid.uuid4())
    await threads_repo.create(request.app.state.app_pool, thread_id, dataset)
    return {"thread_id": thread_id, "dataset": dataset}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request) -> dict:
    """The conversation so far, the SQL a follow-up builds on, and any question waiting for
    a reply (answer it with POST /threads/{id}/resume)."""
    dataset = await _existing_thread_dataset(request, thread_id)
    snapshot = await request.app.state.graph.aget_state(_config(thread_id))
    values = snapshot.values or {}
    messages = [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in values.get("messages", [])
    ]
    return {
        "thread_id": thread_id,
        "dataset": dataset,
        "messages": messages,
        "last_sql": values.get("last_sql"),
        "pending_clarification": await _pending_question(request, thread_id),
        "last_query": await query_log.last_for_thread(request.app.state.app_pool, thread_id),
    }


@router.post("/threads/{thread_id}/ask")
async def ask_stream(thread_id: str, body: AskRequest, request: Request):
    # A new question on a thread that is waiting for a clarification starts a new turn: the
    # pending one is dropped (per_turn_reset clears its fields)
    dataset = await _thread_dataset(request, thread_id)
    inputs = per_turn_reset(dataset, body.question)
    return Turn(request, thread_id, dataset, inputs, body.question, cacheable=True).sse()


@router.post("/threads/{thread_id}/resume")
async def resume_stream(thread_id: str, body: ResumeRequest, request: Request):
    dataset = _check_loaded(request, await _existing_thread_dataset(request, thread_id))
    pending = await _pending_question(request, thread_id)
    if pending is None:
        raise HTTPException(409, "No pending clarification on this thread")
    return Turn(request, thread_id, dataset, Command(resume=body.reply), pending).sse()


@router.post("/ask")
async def ask_once(body: OneShotAskRequest, request: Request) -> dict:
    """Non-streaming convenience (used by evals): runs the same turn, returns the final JSON."""
    if body.thread_id:
        thread_id = body.thread_id
        dataset = await _thread_dataset(request, thread_id)
    else:
        dataset = _resolve_dataset(request, body.dataset)
        thread_id = str(uuid.uuid4())
        await threads_repo.create(request.app.state.app_pool, thread_id, dataset)

    inputs = per_turn_reset(dataset, body.question)
    turn = Turn(request, thread_id, dataset, inputs, body.question, cacheable=True)
    await turn.run()
    if turn.mapper.error:
        raise HTTPException(502, turn.mapper.error)

    state, outcome = turn.state, turn.outcome
    return {
        "thread_id": thread_id,
        "dataset": dataset,
        "query_id": turn.mapper.query_id,
        "outcome": outcome,
        "question": state.get("question"),
        "answer": state.get("answer") if outcome != "clarified" else None,
        "sql": state.get("sql") if outcome in {"answered", "gave_up"} else None,
        "attempts": state.get("attempts", 0),
        "columns": state.get("columns", []),
        "rows": state.get("rows", []),
        "row_count": state.get("row_count", 0),
        "truncated": state.get("truncated", False),
        "chart": state.get("chart"),
        "clarifying_question": turn.mapper.clarifying_question,
        "cached": turn.cached,
        "latency_ms": round((time.perf_counter() - turn.started) * 1000),
    }
