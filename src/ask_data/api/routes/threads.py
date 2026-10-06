import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette import EventSourceResponse

from ask_data.api import threads_repo
from ask_data.api.security import require_api_key
from ask_data.api.streaming import EventMapper, run_turn
from ask_data.config import settings
from ask_data.graph.state import per_turn_reset

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["threads"])


class CreateThreadRequest(BaseModel):
    dataset: str | None = None  # None → settings.default_dataset


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class OneShotAskRequest(AskRequest):
    dataset: str | None = None
    thread_id: str | None = None  # continue an existing thread; default: a new one


def _resolve_dataset(request: Request, name: str | None) -> str:
    name = name or settings.default_dataset
    if not name:
        raise HTTPException(422, "No dataset given and DEFAULT_DATASET is not set")
    if name not in request.app.state.deps.registry:
        raise HTTPException(404, f"Dataset {name!r} is not loaded")
    return name


async def _thread_dataset(request: Request, thread_id: str) -> str:
    pool = request.app.state.app_pool
    dataset = await threads_repo.get_dataset(pool, thread_id)
    if dataset is None:  # lazily bind an unknown thread id to the default dataset
        dataset = await threads_repo.get_or_create(pool, thread_id, _resolve_dataset(request, None))
    if dataset not in request.app.state.deps.registry:
        raise HTTPException(404, f"Dataset {dataset!r} of this thread is no longer loaded")
    return dataset


def _graph_call(request: Request, thread_id: str, dataset: str, question: str):
    config = {"configurable": {"thread_id": thread_id}}
    return request.app.state.graph, per_turn_reset(dataset, question), config


@router.post("/threads")
async def create_thread(body: CreateThreadRequest, request: Request) -> dict:
    dataset = _resolve_dataset(request, body.dataset)
    thread_id = str(uuid.uuid4())
    await threads_repo.create(request.app.state.app_pool, thread_id, dataset)
    return {"thread_id": thread_id, "dataset": dataset}


@router.post("/threads/{thread_id}/ask")
async def ask_stream(thread_id: str, body: AskRequest, request: Request):
    dataset = await _thread_dataset(request, thread_id)
    graph, inputs, config = _graph_call(request, thread_id, dataset, body.question)
    mapper = EventMapper(query_id=str(uuid.uuid4()))

    async def events():
        async for name, data in run_turn(graph, inputs, config, request.app.state.deps, mapper):
            yield {"event": name, "data": json.dumps(data, default=str)}

    return EventSourceResponse(events())


@router.post("/ask")
async def ask_once(body: OneShotAskRequest, request: Request) -> dict:
    """Non-streaming convenience (used by evals): runs the same turn, returns the final JSON."""
    started = time.perf_counter()
    if body.thread_id:
        thread_id = body.thread_id
        dataset = await _thread_dataset(request, thread_id)
    else:
        dataset = _resolve_dataset(request, body.dataset)
        thread_id = str(uuid.uuid4())
        await threads_repo.create(request.app.state.app_pool, thread_id, dataset)

    graph, inputs, config = _graph_call(request, thread_id, dataset, body.question)
    mapper = EventMapper(query_id=str(uuid.uuid4()))
    error = None
    async for name, data in run_turn(graph, inputs, config, request.app.state.deps, mapper):
        if name == "error":
            error = data["message"]
    if error:
        raise HTTPException(502, error)

    state = (await graph.aget_state(config)).values
    return {
        "thread_id": thread_id,
        "dataset": dataset,
        "query_id": mapper.query_id,
        "outcome": state.get("outcome"),
        "answer": state.get("answer"),
        "sql": state.get("sql") if state.get("outcome") != "declined" else None,
        "attempts": state.get("attempts", 0),
        "columns": state.get("columns", []),
        "rows": state.get("rows", []),
        "row_count": state.get("row_count", 0),
        "truncated": state.get("truncated", False),
        "chart": state.get("chart"),
        "clarifying_question": state.get("clarifying_question"),
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
