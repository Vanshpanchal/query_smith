"""Table cards: one short text per table, embedded for schema retrieval (`app.schema_cards`).

A card says what a table holds in words a question would use: the table comment, every column
with its type and comment, what it joins to, and the values of small code columns (statuses,
categories). All of it comes from the catalog and the data, so it works for any dataset.
"""

from psycopg import AsyncConnection
from psycopg import sql as pgsql
from psycopg.rows import tuple_row

from ask_data.schema.catalog import SchemaInfo, TableInfo

SAMPLE_MAX_DISTINCT = 12  # a text column with more distinct values is free text, not a code
TEXT_TYPES = ("text", "character varying", "character")


def build_card(schema: str, table: TableInfo, samples: dict[str, list[str]] | None = None) -> str:
    samples = samples or {}
    parts = [f"table {schema}.{table.name}"]
    if table.comment:
        parts[0] += f" — {' '.join(table.comment.split()).rstrip('.')}"
    columns = []
    for c in table.columns:
        text = f"{c.name} {c.type}"
        if c.comment:
            text += f" ({' '.join(c.comment.split())})"
        columns.append(text)
    parts.append("columns: " + ", ".join(columns))
    if table.references:
        parts.append("joins to: " + ", ".join(sorted(table.references)))
    values = [f"{col} ∈ {{{', '.join(vals)}}}" for col, vals in samples.items() if vals]
    if values:
        parts.append("sample values: " + "; ".join(values))
    return ". ".join(parts)


async def read_samples(
    conn: AsyncConnection, schema: str, table: TableInfo
) -> dict[str, list[str]]:
    """Distinct values of each text column that has at most SAMPLE_MAX_DISTINCT of them."""
    samples: dict[str, list[str]] = {}
    async with conn.cursor(row_factory=tuple_row) as cur:
        for c in table.columns:
            if not c.type.startswith(TEXT_TYPES):
                continue
            query = pgsql.SQL(
                "SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL ORDER BY 1 LIMIT {n}"
            ).format(
                col=pgsql.Identifier(c.name),
                tbl=pgsql.Identifier(schema, table.name),
                n=pgsql.Literal(SAMPLE_MAX_DISTINCT + 1),
            )
            await cur.execute(query)
            values = [str(r[0]) for r in await cur.fetchall()]
            if len(values) <= SAMPLE_MAX_DISTINCT:
                samples[c.name] = values
    return samples


async def build_cards(conn: AsyncConnection, info: SchemaInfo) -> dict[str, str]:
    return {
        name: build_card(info.schema, table, await read_samples(conn, info.schema, table))
        for name, table in info.tables.items()
    }
