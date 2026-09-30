import pytest

from ask_data.datasets.config import IDENT_RE
from ask_data.datasets.loader.sanitize import sanitize_columns, sanitize_identifier


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Order Date (IST)", "order_date_ist"),
        ("CustomerID", "customer_id"),
        ("customerName", "customer_name"),
        ("  amount  ", "amount"),
        ("Café Name", "cafe_name"),
        ("Amount (₹)", "amount"),
        ("% share", "share"),
        ("2024 revenue", "c_2024_revenue"),
        ("", "col"),
        ("!!!", "col"),
        (42, "c_42"),
    ],
)
def test_headers_become_snake_case_identifiers(raw, expected):
    assert sanitize_identifier(raw) == expected


@pytest.mark.parametrize("word", ["order", "user", "over", "group", "select", "table"])
def test_reserved_words_get_a_trailing_underscore(word):
    assert sanitize_identifier(word) == f"{word}_"
    assert sanitize_identifier(word.upper()) == f"{word}_"


def test_long_headers_are_truncated_to_the_identifier_limit():
    name = sanitize_identifier("a very long column header that keeps going and going")
    assert len(name) <= 31
    assert IDENT_RE.match(name)
    assert not name.endswith("_")


def test_duplicates_after_sanitizing_get_numbered_suffixes():
    assert sanitize_columns(["Name", "name", "NAME "]) == ["name", "name_2", "name_3"]


def test_suffix_does_not_collide_with_an_existing_column():
    assert sanitize_columns(["a", "a_2", "a"]) == ["a", "a_2", "a_3"]


def test_suffixed_long_duplicates_still_fit_the_limit():
    long = "x" * 40
    out = sanitize_columns([long, long])
    assert out[0] != out[1]
    assert all(len(n) <= 31 and IDENT_RE.match(n) for n in out)
