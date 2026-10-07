"""Schema retrieval: pick the tables a question needs instead of sending the whole schema.

`index_schema.py` embeds one card per table into `app.schema_cards`; `select_tables` embeds the
question, takes the nearest cards of the thread's dataset (pgvector cosine distance) and adds the
tables needed to join them (`expand_with_fks`). Small datasets skip all of this and send the full
schema (SCHEMA_RETRIEVAL_MIN_TABLES).
"""

import hashlib
import logging
import math
import re
from itertools import combinations
from typing import Protocol

from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

from ask_data.config import settings
from ask_data.schema.catalog import SchemaInfo

log = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # app.schema_cards.embedding is vector(768)
HASHING_MODEL = "hashing"


class Embedder(Protocol):
    model: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class SchemaRetriever(Protocol):
    async def search(self, dataset: str, question: str, k: int) -> list[str]: ...


class HashingEmbedder:
    """Key-free lexical embedder: words (and the parts of snake_case names) hashed into 768
    buckets. It only matches shared words, so it is a fallback and a test double for the pgvector
    path, not a semantic model."""

    model = HASHING_MODEL
    word_re = re.compile(r"[a-z0-9]+")

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        for word in self.word_re.findall(text.lower()):
            for token in {word, word.rstrip("s")}:  # "wickets" ~ "wicket"
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                vec[h % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    def __init__(self, model: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = model
        self._client = GoogleGenerativeAIEmbeddings(model=model)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(
            texts, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBEDDING_DIM
        )

    async def embed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(
            text, task_type="RETRIEVAL_QUERY", output_dimensionality=EMBEDDING_DIM
        )


def build_embedder() -> tuple[Embedder | None, str]:
    """(embedder, status). None when the configured model can't be used; callers then fall back
    to the full schema."""
    model = settings.embedding_model.strip()
    if model == HASHING_MODEL:
        return HashingEmbedder(), "ok (hashing: lexical, no key)"
    if not settings.google_api_key.strip():
        return None, f"embeddings off: GOOGLE_API_KEY is empty (EMBEDDING_MODEL={model})"
    try:
        return GeminiEmbedder(model), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"embeddings off: {type(e).__name__}: {e}"


def to_vector(values: list[float]) -> str:
    """pgvector text form, cast with ::vector (no type adapter needed on the pool)."""
    if len(values) != EMBEDDING_DIM:
        raise ValueError(f"embedding has {len(values)} dimensions, expected {EMBEDDING_DIM}")
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


class PgVectorRetriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder):
        self.pool, self.embedder = pool, embedder

    async def search(self, dataset: str, question: str, k: int) -> list[str]:
        vector = to_vector(await self.embedder.embed_query(question))
        async with self.pool.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(
                "SELECT table_name FROM app.schema_cards "
                "WHERE dataset = %s AND embedding_model = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (dataset, self.embedder.model, vector, k),
            )
            return [r[0] for r in await cur.fetchall()]


def expand_with_fks(selected: list[str], info: SchemaInfo) -> list[str]:
    """`selected` + what's needed to join it: the tables the selected ones reference (names
    behind ids), and one bridge table for any pair that has no direct foreign key."""
    chosen = [t for t in dict.fromkeys(selected) if t in info.tables]
    extra: set[str] = set()
    for t in chosen:
        extra |= info.tables[t].references
    for a, b in combinations(chosen, 2):
        if b in info.neighbours(a):
            continue
        bridges = sorted(info.neighbours(a) & info.neighbours(b))
        if bridges:
            extra.add(bridges[0])
    return chosen + sorted((extra - set(chosen)) & set(info.tables))


async def upsert_cards(
    conn, dataset: str, cards: dict[str, str], vectors: list[list[float]], model: str
) -> None:
    """Replace every card of `dataset` (tables dropped from the dataset lose their card)."""
    async with conn.transaction():
        await conn.execute("DELETE FROM app.schema_cards WHERE dataset = %s", (dataset,))
        for (table, card), vector in zip(cards.items(), vectors, strict=True):
            await conn.execute(
                "INSERT INTO app.schema_cards (dataset, table_name, card, embedding, "
                "embedding_model) VALUES (%s, %s, %s, %s::vector, %s)",
                (dataset, table, card, to_vector(vector), model),
            )
