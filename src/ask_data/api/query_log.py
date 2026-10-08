"""app.query_log (one row per graph run) and app.feedback (👍/👎 on a row)."""

from dataclasses import dataclass

from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool


@dataclass
class QueryLogRow:
    id: str
    thread_id: str
    dataset: str
    question: str
    outcome: str  # answered | clarified | declined | gave_up | error
    final_sql: str | None
    attempts: int
    row_count: int | None
    latency_ms: int
    model: str
    cached: bool = False


async def insert(pool: AsyncConnectionPool, row: QueryLogRow) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO app.query_log (id, thread_id, dataset, question, final_sql, outcome, "
            "attempts, row_count, latency_ms, model, cached) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                row.id, row.thread_id, row.dataset, row.question, row.final_sql, row.outcome,
                row.attempts, row.row_count, row.latency_ms, row.model, row.cached,
            ),
        )  # fmt: skip


async def last_for_thread(pool: AsyncConnectionPool, thread_id: str) -> dict | None:
    async with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            "SELECT id, question, outcome, attempts, latency_ms, created_at FROM app.query_log "
            "WHERE thread_id = %s ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    qid, question, outcome, attempts, latency_ms, created_at = row
    return {"query_id": str(qid), "question": question, "outcome": outcome,
            "attempts": attempts, "latency_ms": latency_ms,
            "created_at": created_at.isoformat()}  # fmt: skip


async def set_feedback(
    pool: AsyncConnectionPool, query_id: str, rating: int, comment: str | None
) -> bool:
    """Create or replace the feedback of a logged query. False if the query id is unknown."""
    async with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            "INSERT INTO app.feedback (query_id, rating, comment) "
            "SELECT id, %s, %s FROM app.query_log WHERE id = %s "
            "ON CONFLICT (query_id) DO UPDATE "
            "SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, updated_at = now() "
            "RETURNING query_id",
            (rating, comment, query_id),
        )
        return await cur.fetchone() is not None
