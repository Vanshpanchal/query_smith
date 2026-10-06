import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ask_data.api.routes import datasets, health, threads
from ask_data.api.security import RateLimiter
from ask_data.config import settings
from ask_data.datasets.registry import DatasetRegistry
from ask_data.graph.builder import build_graph
from ask_data.graph.nodes import AgentDeps
from ask_data.llm import build_structured_llms_or_unavailable, export_provider_env
from ask_data.sql.executor import SqlExecutor, reset_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    export_provider_env()
    # App role: checkpoints, app.threads, registry. AsyncPostgresSaver needs autocommit + dict rows.
    app_pool = AsyncConnectionPool(
        settings.app_database_url,
        open=False,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    # Reader role: the ONLY connection that runs LLM-written SQL. RESET ALL on return, so a
    # session-level set_config(...) that slipped through can't follow the connection around.
    reader_pool = AsyncConnectionPool(
        settings.reader_database_url,
        open=False,
        max_size=5,
        kwargs={"autocommit": True},
        reset=reset_session,
    )
    await app_pool.open(wait=True, timeout=15)
    await reader_pool.open(wait=True, timeout=15)
    redis = aioredis.from_url(settings.redis_url)
    try:
        checkpointer = AsyncPostgresSaver(app_pool)
        await checkpointer.setup()  # creates checkpoint tables (idempotent)
        async with app_pool.connection() as conn:
            registry = await DatasetRegistry.load(settings.datasets_dir, conn)

        llms, app.state.llm_status = build_structured_llms_or_unavailable()
        app.state.deps = AgentDeps(
            registry=registry,
            executor=SqlExecutor(reader_pool, settings.statement_timeout_ms),
            intent_llm=llms["intent"],
            sql_llm=llms["sql"],
            answer_llm=llms["answer"],
        )
        app.state.graph = build_graph(checkpointer)
        app.state.app_pool, app.state.reader_pool, app.state.redis = app_pool, reader_pool, redis
        app.state.rate_limiter = RateLimiter(redis, settings.rate_limit_per_minute)
        yield
    finally:
        await redis.aclose()
        await reader_pool.close()
        await app_pool.close()


app = FastAPI(title="AskData", lifespan=lifespan)
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(datasets.admin_router)
app.include_router(threads.router)
