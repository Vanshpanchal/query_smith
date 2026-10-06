"""app.threads: which dataset each LangGraph thread is bound to (for life)."""

from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool


async def create(pool: AsyncConnectionPool, thread_id: str, dataset: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO app.threads (thread_id, dataset) VALUES (%s, %s)", (thread_id, dataset)
        )


async def get_dataset(pool: AsyncConnectionPool, thread_id: str) -> str | None:
    async with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute("SELECT dataset FROM app.threads WHERE thread_id = %s", (thread_id,))
        row = await cur.fetchone()
        return row[0] if row else None


async def get_or_create(pool: AsyncConnectionPool, thread_id: str, default: str) -> str:
    """The thread's dataset; an unknown thread id is bound to `default` (lazy creation)."""
    async with pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            "INSERT INTO app.threads (thread_id, dataset) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO NOTHING",
            (thread_id, default),
        )
        await cur.execute("SELECT dataset FROM app.threads WHERE thread_id = %s", (thread_id,))
        return (await cur.fetchone())[0]
