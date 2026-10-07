"""Table cards, foreign-key expansion and the key-free embedder (no database)."""

import math

import pytest

from ask_data.schema.cards import build_card
from ask_data.schema.catalog import ColumnInfo, SchemaInfo, TableInfo
from ask_data.schema.retrieval import EMBEDDING_DIM, HashingEmbedder, expand_with_fks, to_vector


def table(name: str, *refs: str, comment: str | None = None) -> TableInfo:
    keys = ["PRIMARY KEY (id)"] + [f"FOREIGN KEY ({r}_id) REFERENCES s.{r}(id)" for r in refs]
    return TableInfo(name, comment, [ColumnInfo("id", "bigint")], keys)


# star schema: facts → (dims); `bridge` links `left` and `right`, which don't reference each other
INFO = SchemaInfo(
    "s",
    {
        "facts": table("facts", "dims", "people"),
        "dims": table("dims"),
        "people": table("people"),
        "left": table("left"),
        "right": table("right"),
        "bridge": table("bridge", "left", "right"),
        "lonely": table("lonely"),
    },
)


def test_references_parse_qualified_unqualified_and_quoted():
    t = TableInfo("t", keys=[
        "FOREIGN KEY (a) REFERENCES ipl.matches(id)",
        "FOREIGN KEY (b) REFERENCES players(id)",
        'FOREIGN KEY (c) REFERENCES "sales"."orders" (order_id)',
        "PRIMARY KEY (id)",
    ])  # fmt: skip
    assert t.references == {"matches", "players", "orders"}


def test_neighbours_are_undirected():
    assert INFO.neighbours("dims") == {"facts"}
    assert INFO.neighbours("facts") == {"dims", "people"}
    assert INFO.neighbours("lonely") == set()


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (["facts"], ["facts", "dims", "people"]),  # names behind the ids come along
        (["dims"], ["dims"]),  # a dimension alone needs nothing else
        (["left", "right"], ["left", "right", "bridge"]),  # no direct FK: add the bridge
        (["facts", "facts", "unknown"], ["facts", "dims", "people"]),  # dedup, drop unknown
        (["lonely", "dims"], ["lonely", "dims"]),  # nothing connects them: leave as is
        ([], []),
    ],
)
def test_expand_with_fks(selected, expected):
    assert expand_with_fks(selected, INFO) == expected


def test_build_card():
    t = TableInfo(
        "orders",
        "One row per order.",
        [ColumnInfo("id", "bigint"), ColumnInfo("status", "text", "Order status")],
        ["PRIMARY KEY (id)", "FOREIGN KEY (customer_id) REFERENCES shop.customers(id)"],
    )
    card = build_card("shop", t, {"status": ["delivered", "returned"]})
    assert card == (
        "table shop.orders — One row per order. columns: id bigint, status text (Order status). "
        "joins to: customers. sample values: status ∈ {delivered, returned}"
    )


async def test_hashing_embedder_ranks_by_shared_words():
    emb = HashingEmbedder()
    docs = await emb.embed_documents(
        ["table s.wickets bowler dismissal kind", "table s.venues ground city name"]
    )
    query = await emb.embed_query("which bowler took the most wickets")
    assert len(query) == EMBEDDING_DIM and math.isclose(sum(v * v for v in query), 1.0)
    scores = [sum(a * b for a, b in zip(query, d, strict=True)) for d in docs]
    assert scores[0] > scores[1]


def test_to_vector_checks_dimensions():
    assert to_vector([0.5] * EMBEDDING_DIM).startswith("[0.5,0.5")
    with pytest.raises(ValueError, match="expected 768"):
        to_vector([1.0, 2.0])
