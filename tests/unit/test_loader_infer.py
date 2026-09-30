from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from ask_data.datasets.loader.infer import (
    ConversionError,
    ParseOptions,
    convert,
    infer,
    to_strings,
    type_family,
)

PLAIN = ParseOptions()
INDIAN = ParseOptions(thousands=",", dayfirst=True)


def _infer(values, opts=PLAIN, threshold=0.99):
    return infer("c", values, opts, threshold)


# --- to_strings ---------------------------------------------------------------------------------


def test_to_strings_strips_and_maps_blanks_and_null_words_to_none():
    s = pd.Series([" a ", "", "   ", None, "N/A", "b"])
    assert to_strings(s, frozenset({"N/A"})) == ["a", None, None, None, None, "b"]


def test_to_strings_turns_integral_floats_back_into_ints():
    # A transform.py int column with a NaN comes out of pandas as float: 1.0, NaN, 2.0
    assert to_strings(pd.Series([1.0, None, 2.0])) == ["1", None, "2"]


def test_to_strings_keeps_real_fractions():
    assert to_strings(pd.Series([1.5, 2.0])) == ["1.5", "2.0"]


# --- inference ----------------------------------------------------------------------------------


def test_infers_bigint():
    col = _infer(["1", "-2", "300"])
    assert (col.pg_type, col.values) == ("bigint", [1, -2, 300])


def test_thousands_separator_is_honoured():
    col = _infer(["1,200", "3,400,000"], INDIAN)
    assert (col.pg_type, col.values) == ("bigint", [1200, 3_400_000])


def test_thousands_separator_without_the_option_stays_text():
    # Regression (Phase 1.3): pandas parsed "1,200" as a date in the year 200
    assert _infer(["1,200", "3,400"]).pg_type == "text"
    assert _infer(["1,200.50", "3,400"]).pg_type == "text"


def test_infers_numeric_as_decimal():
    col = _infer(["1,200.50", "0.5", "7"], INDIAN)
    assert col.pg_type == "numeric"
    assert col.values == [Decimal("1200.50"), Decimal("0.5"), Decimal(7)]


@pytest.mark.parametrize("codes", [["011001", "560001"], ["007", "12"], ["00.5", "1.5"]])
def test_leading_zero_codes_stay_text(codes):
    col = _infer(codes)
    assert col.pg_type == "text"
    assert col.values == codes


def test_day_first_date():
    col = _infer(["27/09/2026", "01/02/2025"], INDIAN)
    assert col.pg_type == "date"
    assert col.values == [date(2026, 9, 27), date(2025, 2, 1)]


def test_month_first_by_default():
    col = _infer(["01/02/2025"])
    assert col.values == [date(2025, 1, 2)]


def test_iso_dates_are_not_swapped_by_dayfirst():
    # Regression (Phase 1.2): with dayfirst=True pandas read 2025-10-01 as 10 January
    col = _infer(["2025-10-01 09:30", "2025-10-02 18:00"], INDIAN)
    assert col.pg_type == "timestamp"
    assert col.values == [datetime(2025, 10, 1, 9, 30), datetime(2025, 10, 2, 18, 0)]  # noqa: DTZ001


def test_iso_date_only_mixed_with_dayfirst_strings():
    col = _infer(["2025-10-01", "27/09/2026"], INDIAN)
    assert col.values == [date(2025, 10, 1), date(2026, 9, 27)]


def test_timezone_offset_gives_timestamptz():
    assert _infer(["2025-10-01T09:30:00+05:30", "2025-10-02T10:00:00+05:30"]).pg_type == (
        "timestamptz"
    )


def test_bare_month_names_are_not_dates():
    assert _infer(["May", "June"]).pg_type == "text"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["yes", "no", "YES", "No"], [True, False, True, False]),
        (["true", "false"], [True, False]),
        (["Y", "n"], [True, False]),
    ],
)
def test_infers_boolean_words(raw, expected):
    col = _infer(raw)
    assert (col.pg_type, col.values) == ("boolean", expected)


def test_zero_one_column_is_a_number_not_a_boolean():
    # bigint is tried first; a 0/1 column only becomes boolean if the yaml says so
    assert _infer(["0", "1", "1"]).pg_type == "bigint"


def test_all_null_column_is_text():
    col = _infer([None, None])
    assert (col.pg_type, col.values) == ("text", [None, None])


def test_nulls_are_preserved():
    assert _infer(["1", None, "3"]).values == [1, None, 3]


def test_values_below_threshold_become_null_and_are_reported():
    values = [str(i) for i in range(199)] + ["n/a-ish"]
    col = _infer(values, threshold=0.99)
    assert col.pg_type == "bigint"
    assert col.values[-1] is None
    assert (col.nulled, col.nulled_examples) == (1, ["n/a-ish"])


def test_too_many_misfits_fall_back_to_text():
    col = _infer(["1", "2", "three"], threshold=0.99)
    assert (col.pg_type, col.nulled) == ("text", 0)


# --- declared (yaml) types are strict -----------------------------------------------------------


def test_convert_to_declared_type():
    col = convert("amount", ["1,200.50", None], "numeric(12,2)", INDIAN)
    assert (col.pg_type, col.inferred) == ("numeric(12,2)", False)
    assert col.values == [Decimal("1200.50"), None]


def test_convert_zero_one_to_boolean():
    assert convert("flag", ["0", "1"], "boolean", PLAIN).values == [False, True]


def test_convert_rejects_a_single_bad_value_with_examples():
    with pytest.raises(ConversionError, match=r"column 'amount': 1 value\(s\).*'abc'"):
        convert("amount", ["1", "abc", "3"], "numeric(12,2)", PLAIN)


def test_convert_int_enforces_32_bit_range():
    with pytest.raises(ConversionError):
        convert("n", ["2147483648"], "int", PLAIN)
    assert convert("n", ["2147483648"], "bigint", PLAIN).values == [2_147_483_648]


def test_convert_date_drops_time_of_day():
    assert convert("d", ["2025-10-01"], "date", PLAIN).values == [date(2025, 10, 1)]


@pytest.mark.parametrize(
    ("a", "b"),
    [("int", "bigint"), ("numeric(12,2)", "double precision"), ("timestamp", "timestamptz")],
)
def test_type_families_group_joinable_types(a, b):
    assert type_family(a) == type_family(b)


def test_integer_and_text_are_different_families():
    assert type_family("bigint") != type_family("text")
