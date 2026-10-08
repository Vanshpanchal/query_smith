"""Read a dataset schema from the Postgres catalog and render it for the SQL prompt.

The catalog, not dataset.yaml, is the source of truth: it is what the SQL runs against, and the
loader already wrote every yaml comment into it with COMMENT ON. So one function works for any
dataset (moved here from notebooks/v0_sql_agent.ipynb).
"""

import re
from dataclasses import dataclass, field

from psycopg import AsyncConnection
from psycopg.rows import tuple_row

COLUMNS_SQL = """
SELECT c.relname, obj_description(c.oid, 'pg_class'),
       a.attname, format_type(a.atttypid, a.atttypmod), col_description(c.oid, a.attnum)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE n.nspname = %(schema)s AND c.relkind = 'r'
ORDER BY c.relname, a.attnum
"""
KEYS_SQL = """
SELECT cl.relname, con.contype, pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class cl ON cl.oid = con.conrelid
JOIN pg_namespace n ON n.oid = con.connamespace
WHERE n.nspname = %(schema)s AND con.contype IN ('p', 'f')
ORDER BY 1, 2 DESC, 3
"""
# pg_get_constraintdef: "FOREIGN KEY (a) REFERENCES myschema.matches(id)" (schema-qualified unless the
# schema is on the search_path) → "matches"
FK_TARGET_RE = re.compile(r'REFERENCES\s+(?:"?\w+"?\.)?"?(\w+)"?\s*\(', re.IGNORECASE)


@dataclass
class ColumnInfo:
    name: str
    type: str
    comment: str | None = None


@dataclass
class TableInfo:
    name: str
    comment: str | None = None
    columns: list[ColumnInfo] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)  # "PRIMARY KEY (id)", "FOREIGN KEY ..."

    def render(self, schema: str) -> str:
        lines = [f"-- {' '.join(self.comment.split())}"] if self.comment else []
        lines.append(f"TABLE {schema}.{self.name} (")
        for c in self.columns:
            line = f"  {c.name} {c.type}"
            lines.append(f"{line}  -- {' '.join(c.comment.split())}" if c.comment else line)
        lines += [f"  {k}" for k in self.keys]
        return "\n".join(lines) + "\n)"

    @property
    def references(self) -> set[str]:
        """Tables this table's foreign keys point to."""
        return {m.group(1) for k in self.keys for m in [FK_TARGET_RE.search(k)] if m}

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "comment": self.comment,
            "columns": [
                {"name": c.name, "type": c.type, "comment": c.comment} for c in self.columns
            ],
            "keys": self.keys,
        }


@dataclass
class SchemaInfo:
    schema: str
    tables: dict[str, TableInfo]

    def render(self, tables: list[str] | None = None) -> str:
        """Commented CREATE TABLE-like blocks for `tables` (default: all), in catalog order."""
        names = [t for t in self.tables if tables is None or t in tables]
        return "\n\n".join(self.tables[t].render(self.schema) for t in names)

    def neighbours(self, table: str) -> set[str]:
        """Tables joined to `table` by a foreign key, in either direction."""
        out = self.tables[table].references if table in self.tables else set()
        into = {name for name, t in self.tables.items() if table in t.references}
        return (out | into) & set(self.tables) - {table}


async def read_schema(conn: AsyncConnection, schema: str) -> SchemaInfo:
    tables: dict[str, TableInfo] = {}
    async with conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(COLUMNS_SQL, {"schema": schema})
        for table, table_comment, column, dtype, comment in await cur.fetchall():
            info = tables.setdefault(table, TableInfo(table, table_comment))
            info.columns.append(ColumnInfo(column, dtype, comment))
        await cur.execute(KEYS_SQL, {"schema": schema})
        for table, _contype, definition in await cur.fetchall():
            if table in tables:
                tables[table].keys.append(definition)
    return SchemaInfo(schema, tables)
