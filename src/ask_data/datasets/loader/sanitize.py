"""Turn arbitrary source column headers into safe, readable Postgres identifiers.

"Order Date (IST)" -> order_date_ist, duplicates get _2/_3, reserved words get a trailing
underscore (order -> order_). Every result matches config.IDENT_RE, so it can go through
sql.Identifier without surprises and be referenced from dataset.yaml by its clean name.
"""

import re
import unicodedata

from ask_data.datasets.config import IDENT_RE

MAX_LEN = 31  # same limit as config.IDENT

# Words an LLM would have to double-quote in generated SQL. Postgres reserved keywords plus a
# few non-reserved ones that routinely break hand-written queries (user, over, year...).
RESERVED_WORDS = frozenset(
    """
    all analyse analyze and any array as asc asymmetric authorization between binary both case
    cast check collate column concurrently constraint create cross current_catalog current_date
    current_role current_schema current_time current_timestamp current_user default deferrable
    desc distinct do else end except false fetch filter for foreign freeze from full grant group
    having ilike in initially inner intersect into is isnull join lateral leading left like limit
    localtime localtimestamp natural not notnull null offset on only or order outer over overlaps
    placing primary references returning right select session_user similar some symmetric table
    tablesample then to trailing true union unique user using variadic verbose when where window
    with
    """.split()  # noqa: SIM905
)


def sanitize_identifier(raw: object) -> str:
    """Best-effort snake_case identifier for one header. Not deduplicated."""
    text = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)  # camelCase -> camel_Case
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not text:
        text = "col"
    if not text[0].isalpha():
        text = f"c_{text}"
    text = text[:MAX_LEN].rstrip("_")
    if text in RESERVED_WORDS:
        text = f"{text}_"
    return text


def sanitize_columns(headers: list[object]) -> list[str]:
    """Sanitize every header and make the results unique: a, a -> a, a_2."""
    seen: set[str] = set()
    out: list[str] = []
    for header in headers:
        base = sanitize_identifier(header)
        name, n = base, 1
        while name in seen:
            n += 1
            suffix = f"_{n}"
            name = f"{base[: MAX_LEN - len(suffix)]}{suffix}"
        seen.add(name)
        out.append(name)
    assert all(IDENT_RE.match(n) for n in out), out
    return out
