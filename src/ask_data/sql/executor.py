"""Run guarded SQL on the read-only reader pool (layers 3 and 5 of the defence in depth).

Every call is its own transaction so SET LOCAL never leaks to the next pool user:
  SET TRANSACTION READ ONLY (must be first) → SET LOCAL statement_timeout → SET LOCAL search_path
The search_path is built with sql.Identifier, never an f-string. `query` has already passed
guard(), and EXPLAIN without ANALYZE never executes it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool


class QueryError(Exception):
    """A database error worded for the LLM to repair the query."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]  # JSON-safe values
    truncated: bool  # more rows existed than max_rows


class Executor(Protocol):
    async def explain_cost(self, query: str, schema: str) -> float: ...
    async def run_select(self, query: str, schema: str, max_rows: int) -> QueryResult: ...


class SqlExecutor:
    def __init__(self, pool: AsyncConnectionPool, statement_timeout_ms: int):
        self.pool = pool
        self.statement_timeout_ms = statement_timeout_ms

    async def explain_cost(self, query: str, schema: str) -> float:
        async with self._read_only_tx(schema) as cur:
            await cur.execute(f"EXPLAIN (FORMAT JSON) {query}")  # type: ignore[arg-type]
            plan = (await cur.fetchone())[0]
            return float(plan[0]["Plan"]["Total Cost"])

    async def run_select(self, query: str, schema: str, max_rows: int) -> QueryResult:
        async with self._read_only_tx(schema) as cur:
            await cur.execute(query)  # type: ignore[arg-type]
            if cur.description is None:
                raise QueryError("The statement returned no result set; write a SELECT")
            rows = await cur.fetchmany(max_rows + 1)
            return QueryResult(
                columns=[d.name for d in cur.description],
                rows=[[json_safe(v) for v in r] for r in rows[:max_rows]],
                truncated=len(rows) > max_rows,
            )

    @asynccontextmanager
    async def _read_only_tx(self, schema: str) -> AsyncIterator[psycopg.AsyncCursor]:
        """A cursor inside a fresh read-only transaction; psycopg errors become QueryError."""
        try:
            async with self.pool.connection() as conn, conn.transaction():
                cur = conn.cursor(row_factory=tuple_row)
                await cur.execute("SET TRANSACTION READ ONLY")
                await cur.execute(
                    pgsql.SQL("SET LOCAL statement_timeout = {}").format(
                        pgsql.Literal(self.statement_timeout_ms)
                    )
                )
                await cur.execute(
                    pgsql.SQL("SET LOCAL search_path = {}").format(pgsql.Identifier(schema))
                )
                yield cur
        except psycopg.Error as e:
            raise QueryError(describe_db_error(e, self.statement_timeout_ms)) from e


async def reset_session(conn: psycopg.AsyncConnection) -> None:
    """Pool `reset` hook: back to the role's defaults (read-only, timeout, empty search_path)."""
    await conn.execute("RESET ALL")


def describe_db_error(e: psycopg.Error, timeout_ms: int) -> str:
    if isinstance(e, psycopg.errors.QueryCanceled):
        return (
            f"Query timed out after {timeout_ms} ms. Simplify it: filter earlier, aggregate "
            "before joining, and avoid joining large tables without a join condition."
        )
    message = str(e).strip()
    diag = e.diag
    hint = f"\nHINT: {diag.message_hint}" if diag and diag.message_hint else ""
    return f"{type(e).__name__}: {message}{hint}"


def json_safe(v: Any) -> Any:
    """Make a DB value JSON- and checkpoint-friendly (Architecture gotcha #5)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() and abs(v) < 2**53 else float(v)
    if isinstance(v, (date, datetime, time)):
        return v.isoformat()
    if isinstance(v, timedelta):
        return str(v)
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    return str(v)
