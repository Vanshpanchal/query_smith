"""SQL guard: layer 2 of the defence in depth (vault Architecture → "SQL Guard").

Parses LLM-written SQL with sqlglot and only lets through ONE read-only query over the thread's
own dataset schema. Error messages go back to the LLM as `error`, so they say what to fix.
The database role (layer 4) still blocks everything on its own; this layer is not the only wall.
"""

import sqlglot
from sqlglot import exp

FORBIDDEN_FUNCS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file",
    "pg_ls_dir", "pg_stat_file", "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "dblink", "dblink_exec", "lo_import", "lo_export", "lo_get", "set_config",
    "current_setting", "query_to_xml", "table_to_xml", "txid_current",
}  # fmt: skip
# Whole families: server files, admin, locks, large objects, remote connections
FORBIDDEN_FUNC_PREFIXES = ("pg_", "lo_", "dblink")
# Harmless pg_* functions an analytics query may legitimately use
ALLOWED_PG_FUNCS = {"pg_typeof"}

FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.Command, exp.Into, exp.Copy, exp.Set, exp.Grant, exp.Transaction, exp.Lock,
)  # fmt: skip


class UnsafeSQL(ValueError):
    """The query is not allowed; str(e) is written for the LLM to act on."""


def guard(sql: str, schema: str, allowed_tables: set[str], max_rows: int) -> str:
    """Return the normalized query (LIMIT added if missing) or raise UnsafeSQL."""
    schema = schema.lower()
    allowed = {t.lower() for t in allowed_tables}
    try:
        trees = [t for t in sqlglot.parse(sql, read="postgres") if t is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"Parse error: {_first_line(e)}") from e
    if len(trees) != 1:
        raise UnsafeSQL("Exactly one statement is allowed (no ';'-separated statements)")
    tree = trees[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQL("Only SELECT queries are allowed")
    if tree.find(*FORBIDDEN_NODES):
        raise UnsafeSQL(
            "Statement contains a forbidden operation (only plain SELECT is allowed; "
            "no INTO, FOR UPDATE, DML or DDL)"
        )

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            continue  # table function, e.g. generate_series(...): checked as a function below
        name, tschema = t.name.lower(), (t.db or schema).lower()
        if t.catalog:
            raise UnsafeSQL(f"Cross-database reference not allowed: {t.sql(dialect='postgres')}")
        if name in ctes and not t.db:
            continue  # a schema-qualified name is never a CTE
        if tschema != schema:
            raise UnsafeSQL(
                f"Schema {tschema!r} is not allowed; only tables in schema {schema!r} can be used"
            )
        if name not in allowed:
            raise UnsafeSQL(
                f"Unknown table: {schema}.{name}. Available tables: {', '.join(sorted(allowed))}"
            )

    for f in tree.find_all(exp.Func):
        fname = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if fname in ALLOWED_PG_FUNCS:
            continue
        if fname in FORBIDDEN_FUNCS or fname.startswith(FORBIDDEN_FUNC_PREFIXES):
            raise UnsafeSQL(f"Function not allowed: {fname}")

    if not tree.args.get("limit"):
        tree = tree.limit(max_rows)
    return tree.sql(dialect="postgres")


def _first_line(e: Exception) -> str:
    return str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
