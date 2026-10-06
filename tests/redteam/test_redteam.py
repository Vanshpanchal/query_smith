"""Red-team suite (vault Testing.md): each attack must be stopped by the guard AND, independently,
by the database role. Neither layer may rely on the other.

(a) guard side: guard() raises; the cartesian join (#10) is a valid SELECT, so the validation layer
    that stops it is the EXPLAIN cost cap.
(b) DB side: the payload runs on a raw reader connection with only the role's defaults (no guard,
    no executor). It must fail, or for payloads that only change session settings (#8, #9), every
    write attempted afterwards in that session must fail.
"""

import psycopg
import pytest
from conftest import requires_db
from psycopg_pool import AsyncConnectionPool

from ask_data.config import settings
from ask_data.sql.executor import SqlExecutor
from ask_data.sql.guard import UnsafeSQL, guard

SCHEMA = "ipl"
TABLES = {"teams", "venues", "players", "matches", "player_match", "deliveries"}
CARTESIAN = "SELECT * FROM ipl.deliveries a, ipl.deliveries b"

PAYLOADS = {
    1: "DROP TABLE ipl.matches",
    2: "UPDATE ipl.teams SET name = 'x'",
    3: "SELECT 1; DELETE FROM ipl.teams",
    4: "SELECT * FROM app.query_log",
    5: "SELECT * FROM checkpoints",
    6: "SELECT pg_sleep(30)",
    7: "SELECT pg_read_file('/etc/passwd')",
    8: "SELECT set_config('default_transaction_read_only', 'off', false)",
    9: "SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE",
    10: CARTESIAN,
    11: "CREATE TEMP TABLE t AS SELECT 1",
    12: "COPY (SELECT 1) TO PROGRAM 'id'",
}
SESSION_ONLY = {8, 9}  # succeed as statements; must not unlock any write
WRITE_PROBES = [
    "CREATE TEMP TABLE pwned AS SELECT 1",
    "CREATE TABLE ipl.pwned (x int)",
    "UPDATE ipl.teams SET name = name",
    "INSERT INTO ipl.venues (id, name) VALUES (-1, 'x')",
]


@pytest.mark.parametrize("n", [n for n in PAYLOADS if n != 10])
def test_guard_blocks(n):
    with pytest.raises(UnsafeSQL):
        guard(PAYLOADS[n], SCHEMA, TABLES, settings.max_rows)


@requires_db
@pytest.mark.parametrize(
    "sql, blocked_by_cost",
    [
        # guard() appends LIMIT 1000: the planner stops after 1000 rows, so it is cheap and harmless
        (CARTESIAN, False),
        # an aggregate must read the whole cross product: the cost cap rejects it before execution
        ("SELECT COUNT(*) FROM ipl.deliveries a, ipl.deliveries b", True),
    ],
)
async def test_cost_cap_on_cartesian(sql, blocked_by_cost):
    query = guard(sql, SCHEMA, TABLES, settings.max_rows)  # both pass the parser
    async with AsyncConnectionPool(
        settings.reader_database_url, kwargs={"autocommit": True}, open=False
    ) as pool:
        cost = await SqlExecutor(pool, settings.statement_timeout_ms).explain_cost(query, SCHEMA)
    assert (cost > settings.max_plan_cost) is blocked_by_cost


def _reader() -> psycopg.Connection:
    return psycopg.connect(settings.reader_database_url, autocommit=True)


def _fails(conn: psycopg.Connection, sql: str) -> bool:
    try:
        cur = conn.execute(sql)  # type: ignore[arg-type]
        if cur.description:
            cur.fetchall()
    except psycopg.Error:
        return True
    return False


@requires_db
@pytest.mark.parametrize("n", sorted(PAYLOADS))
def test_db_role_blocks(n):
    with _reader() as conn:
        if n in SESSION_ONLY:
            conn.execute(PAYLOADS[n])  # allowed: it only changes this session's defaults
            for probe in WRITE_PROBES:
                assert _fails(conn, probe), f"#{n} unlocked a write: {probe}"
        else:
            assert _fails(conn, PAYLOADS[n]), f"#{n} was not blocked by the database"


@requires_db
def test_reader_role_defaults():
    with _reader() as conn:
        assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
        assert conn.execute("SHOW statement_timeout").fetchone()[0] == "5s"
        privileges = conn.execute(
            "SELECT has_database_privilege(current_user, current_database(), 'TEMP'),"
            "       has_database_privilege(current_user, current_database(), 'CREATE'),"
            "       has_schema_privilege(current_user, 'public', 'USAGE')"
        ).fetchone()
        assert privileges == (False, False, False)
