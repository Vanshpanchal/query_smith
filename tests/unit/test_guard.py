"""guard(): every case from vault Testing.md → "Unit — guard cases", plus a few extras."""

import pytest

from ask_data.sql.guard import UnsafeSQL, guard

TABLES = {"teams", "matches", "deliveries"}


def ok(sql: str, schema: str = "ipl", tables: set[str] = TABLES, max_rows: int = 1000) -> str:
    return guard(sql, schema, tables, max_rows)


def blocked(sql: str, schema: str = "ipl", tables: set[str] = TABLES) -> str:
    with pytest.raises(UnsafeSQL) as e:
        guard(sql, schema, tables, 1000)
    return str(e.value)


def test_plain_select_gets_limit():
    assert ok("SELECT name FROM teams") == "SELECT name FROM teams LIMIT 1000"


def test_schema_qualified_own_schema_passes():
    assert ok("SELECT t.name FROM ipl.teams AS t").endswith("LIMIT 1000")


def test_cte_name_is_not_a_table():
    out = ok("WITH w AS (SELECT name FROM teams) SELECT * FROM w")
    assert "LIMIT 1000" in out


def test_union_passes_and_is_limited():
    out = ok("SELECT name FROM teams UNION SELECT name FROM teams")
    assert out.endswith("LIMIT 1000")


def test_existing_limit_untouched():
    assert ok("SELECT name FROM teams LIMIT 10") == "SELECT name FROM teams LIMIT 10"


def test_trailing_semicolon_is_one_statement():
    assert ok("SELECT name FROM teams;").startswith("SELECT name FROM teams")


def test_table_function_allowed():
    ok("SELECT g FROM generate_series(1, 10) AS g")


def test_pg_typeof_allowed():
    ok("SELECT pg_typeof(name) FROM teams")


def test_delete_blocked():
    assert "Only SELECT" in blocked("DELETE FROM matches")


def test_multi_statement_blocked():
    assert "one statement" in blocked("SELECT 1; DROP TABLE matches")


def test_other_schema_blocked():
    assert "'app'" in blocked("SELECT * FROM app.query_log")


def test_other_dataset_schema_blocked():
    # an ipl table on a sales thread
    assert "'ipl'" in blocked("SELECT * FROM ipl.matches", schema="sales", tables={"orders"})


def test_schema_qualified_name_is_never_a_cte():
    blocked("WITH orders AS (SELECT 1) SELECT * FROM app.orders")


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM pg_catalog.pg_user", "SELECT * FROM pg_shadow", "SELECT * FROM pg_user"]
)
def test_catalog_blocked(sql):
    blocked(sql)


def test_unknown_table_lists_available():
    msg = blocked("SELECT * FROM team")
    assert "Unknown table" in msg and "teams" in msg


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10)",
        "SELECT pg_catalog.pg_sleep(1)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM pg_ls_dir('.')",
        "SELECT set_config('default_transaction_read_only', 'off', false)",
        "SELECT current_setting('is_superuser')",
        "SELECT dblink('host=x', 'select 1')",
        "SELECT lo_import('/etc/passwd')",
    ],
)
def test_dangerous_functions_blocked(sql):
    assert "Function not allowed" in blocked(sql)


def test_select_into_blocked():
    blocked("SELECT * INTO newtab FROM teams")


def test_for_update_blocked():
    blocked("SELECT name FROM teams FOR UPDATE")


def test_data_modifying_cte_blocked():
    blocked("WITH x AS (DELETE FROM teams RETURNING *) SELECT * FROM x")


@pytest.mark.parametrize(
    "sql",
    [
        "COPY teams TO '/tmp/x'",
        "COPY (SELECT 1) TO PROGRAM 'id'",
        "SET search_path = app",
        "SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE",
        "CREATE TEMP TABLE t AS SELECT 1",
        "GRANT SELECT ON teams TO public",
        "BEGIN",
        "EXPLAIN SELECT 1",
        "VALUES (1)",
    ],
)
def test_non_select_statements_blocked(sql):
    blocked(sql)


def test_parse_error_message_for_llm():
    assert blocked("SELEC name FROM teams").startswith("Parse error")


def test_case_insensitive_names():
    ok("SELECT Name FROM IPL.Teams")
