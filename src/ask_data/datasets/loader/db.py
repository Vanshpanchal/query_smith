"""Write prepared tables into Postgres: one dataset = one schema, all in ONE transaction.

If anything fails, the transaction rolls back and the previously loaded version of the
schema (dropped at the start of the transaction under mode=replace) is still there.
Every identifier goes through sql.Identifier and every comment through sql.Literal —
COMMENT ON can't take bind parameters, and nothing from yaml is ever f-stringed into SQL.
"""

from dataclasses import dataclass

import psycopg
from psycopg import sql

from ask_data.datasets.config import DatasetCfg, is_allowed_type
from ask_data.datasets.loader.prepare import LoadError, PreparedTable


@dataclass
class LoadResult:
    schema: str
    table_count: int
    row_count: int
    version: str


def write_dataset(
    conn: psycopg.Connection,
    cfg: DatasetCfg,
    tables: dict[str, PreparedTable],
    version: str,
    reader_role: str,
) -> LoadResult:
    schema = cfg.schema_name
    with conn.transaction(), conn.cursor() as cur:
        exists = cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)).fetchone()
        if exists and cfg.load.mode == "fail_if_exists":
            raise LoadError([f"schema {schema!r} already exists and load.mode is fail_if_exists"])
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

        for table in tables.values():
            _create_and_copy(cur, schema, table)
        # Constraints after COPY: faster, and data problems were already reported in prepare
        for table in tables.values():
            _add_keys(cur, schema, table)
        for table_name, cols in cfg.parsed_indexes:
            cur.execute(
                sql.SQL("CREATE INDEX ON {} ({})").format(
                    _qualified(schema, table_name), _idents(cols)
                )
            )
        for table in tables.values():
            _add_comments(cur, schema, table)

        cur.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(reader_role)
            )
        )
        cur.execute(
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(reader_role)
            )
        )
        for table in tables.values():
            cur.execute(sql.SQL("ANALYZE {}").format(_qualified(schema, table.name)))

        row_count = sum(t.row_count for t in tables.values())
        cur.execute(
            """
            INSERT INTO app.datasets (name, schema_name, title, version, table_count, row_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
              schema_name = EXCLUDED.schema_name, title = EXCLUDED.title,
              version = EXCLUDED.version, table_count = EXCLUDED.table_count,
              row_count = EXCLUDED.row_count, loaded_at = now()
            """,
            (cfg.name, schema, cfg.title, version, len(tables), row_count),
        )
    return LoadResult(schema=schema, table_count=len(tables), row_count=row_count, version=version)


def _create_and_copy(cur: psycopg.Cursor, schema: str, table: PreparedTable) -> None:
    col_defs = []
    for c in table.columns:
        # pg_type is a validated yaml type or one of infer's fixed outputs: safe to inline
        assert is_allowed_type(c.pg_type), c.pg_type
        col_defs.append(sql.SQL("{} {}").format(sql.Identifier(c.name), sql.SQL(c.pg_type)))
    target = _qualified(schema, table.name)
    cur.execute(sql.SQL("CREATE TABLE {} ({})").format(target, sql.SQL(", ").join(col_defs)))
    names = table.column_names()
    with cur.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(target, _idents(names))) as copy:
        for row in table.rows(names):
            copy.write_row(row)


def _add_keys(cur: psycopg.Cursor, schema: str, table: PreparedTable) -> None:
    target = _qualified(schema, table.name)
    if table.cfg.primary_key:
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(
                target, _idents(table.cfg.primary_key)
            )
        )
    for fk in table.cfg.foreign_keys:
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD FOREIGN KEY ({}) REFERENCES {} ({})").format(
                target,
                _idents(fk.columns),
                _qualified(schema, fk.ref_table),
                _idents(fk.ref_columns),
            )
        )


def _add_comments(cur: psycopg.Cursor, schema: str, table: PreparedTable) -> None:
    target = _qualified(schema, table.name)
    if table.cfg.comment:
        cur.execute(
            sql.SQL("COMMENT ON TABLE {} IS {}").format(
                target, sql.Literal(table.cfg.comment.strip())
            )
        )
    for col_name, col_cfg in table.cfg.columns.items():
        if col_cfg.comment:
            cur.execute(
                sql.SQL("COMMENT ON COLUMN {} IS {}").format(
                    sql.Identifier(schema, table.name, col_name),
                    sql.Literal(col_cfg.comment.strip()),
                )
            )


def _qualified(schema: str, table: str) -> sql.Identifier:
    return sql.Identifier(schema, table)


def _idents(names: list[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(n) for n in names)
