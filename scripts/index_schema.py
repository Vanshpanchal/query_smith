"""Build one table card per table of a loaded dataset, embed it and store it in app.schema_cards.

    uv run python scripts/index_schema.py ipl                 # EMBEDDING_MODEL from .env
    uv run python scripts/index_schema.py ipl --print         # show the cards, write nothing
    uv run python scripts/index_schema.py ipl --model hashing # key-free lexical vectors

Runs at the end of `make load`. Without a usable embedder it prints why and exits 0: the agent
then sends the full schema, which is what datasets up to SCHEMA_RETRIEVAL_MIN_TABLES get anyway.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import psycopg
from psycopg.rows import tuple_row

from ask_data.config import settings
from ask_data.datasets.config import load_dataset_cfg
from ask_data.schema.cards import build_cards
from ask_data.schema.catalog import read_schema
from ask_data.schema.retrieval import HASHING_MODEL, HashingEmbedder, build_embedder, upsert_cards


async def run(name: str, model: str | None, print_only: bool) -> int:
    cfg = load_dataset_cfg(Path(settings.datasets_dir) / name)
    async with await psycopg.AsyncConnection.connect(
        settings.app_database_url, autocommit=True
    ) as conn:
        async with conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute("SELECT 1 FROM app.datasets WHERE name = %s", (name,))
            if await cur.fetchone() is None:
                print(f"✗ {name}: not loaded (run make load DATASET={name} first)", file=sys.stderr)
                return 1
        info = await read_schema(conn, cfg.schema_name)
        cards = await build_cards(conn, info)
        if print_only:
            for card in cards.values():
                print(card, end="\n\n")
            return 0

        if model == HASHING_MODEL:
            embedder, status = HashingEmbedder(), "ok"
        else:
            embedder, status = build_embedder()
        if embedder is None:
            print(f"⚠ {name}: schema cards not indexed ({status}); the agent sends the full schema")
            return 0
        vectors = await embedder.embed_documents(list(cards.values()))
        await upsert_cards(conn, name, cards, vectors, embedder.model)
    print(f"✓ {name}: {len(cards)} table card(s) indexed with {embedder.model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="dataset name (loaded with make load)")
    parser.add_argument("--model", help=f"override EMBEDDING_MODEL, e.g. {HASHING_MODEL}")
    parser.add_argument("--print", action="store_true", help="print the cards, write nothing")
    args = parser.parse_args()
    return asyncio.run(run(args.name, args.model, args.print))


if __name__ == "__main__":
    sys.exit(main())
