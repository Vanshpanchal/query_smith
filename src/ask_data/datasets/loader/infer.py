"""Column type inference and value conversion.

Every column (CSV, parquet or a transform.py DataFrame) is first normalized to a list of
stripped strings / None, then either
  * converted to the yaml override type (strict: any unparseable value is an error), or
  * inferred: the first of bigint, numeric, date/timestamp, boolean that parses
    >= threshold of the non-null values wins; the few values that don't fit become NULL and are reported.

Values come out as plain Python objects (int, Decimal, bool, date, datetime, str) because
psycopg's COPY adapts those natively — numpy scalars it does not.
"""

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import pandas as pd

INT32 = (-(2**31), 2**31 - 1)
INT64 = (-(2**63), 2**63 - 1)
INT_RE = re.compile(r"^[+-]?\d+$")
LEADING_ZERO_RE = re.compile(r"^[+-]?0\d")  # 007, 00123: codes, not numbers
DATE_SHAPE_RE = re.compile(r"\d+\D+\d+")  # needs two digit groups; "May" alone is not a date
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
TRUE_WORDS = {"true", "t", "yes", "y", "1"}
FALSE_WORDS = {"false", "f", "no", "n", "0"}

_FAIL = object()


@dataclass
class ParseOptions:
    thousands: str | None = None
    dayfirst: bool = False


@dataclass
class Column:
    name: str
    pg_type: str
    values: list  # Python values, None for NULL
    inferred: bool
    nulled: int = 0  # non-null source values that did not fit the inferred type
    nulled_examples: list[str] = field(default_factory=list)


class ConversionError(ValueError):
    pass


def type_family(pg_type: str) -> str:
    """Types that can be joined / FK-referenced against each other without a cast."""
    if pg_type in {"int", "bigint"}:
        return "integer"
    if pg_type.startswith("numeric") or pg_type == "double precision":
        return "numeric"
    if pg_type in {"timestamp", "timestamptz"}:
        return "timestamp"
    return pg_type


def to_strings(series: pd.Series, null_values: frozenset[str] = frozenset()) -> list[str | None]:
    """Normalize any pandas column to stripped strings, None for NULL/blank."""
    if pd.api.types.is_float_dtype(series):
        nonnull = series.dropna()
        if len(nonnull) and (nonnull == nonnull.round()).all():
            series = series.astype("Int64")  # 3.0 -> 3: int ids that pandas widened for NaN
    nulls = series.isna().tolist()
    out: list[str | None] = []
    for v, is_null in zip(series.tolist(), nulls, strict=True):
        if is_null:
            out.append(None)
            continue
        s = str(v).strip()
        out.append(None if s == "" or s in null_values else s)
    return out


# --- per-type parsers: str -> value | _FAIL --------------------------------------------------


def _int_parser(bounds: tuple[int, int], opts: ParseOptions) -> Callable[[str], object]:
    def parse(s: str) -> object:
        if opts.thousands:
            s = s.replace(opts.thousands, "")
        if not INT_RE.match(s) or LEADING_ZERO_RE.match(s):
            return _FAIL
        n = int(s)
        return n if bounds[0] <= n <= bounds[1] else _FAIL

    return parse


def _numeric_parser(opts: ParseOptions) -> Callable[[str], object]:
    def parse(s: str) -> object:
        if opts.thousands:
            s = s.replace(opts.thousands, "")
        if LEADING_ZERO_RE.match(s):  # "0.5" is fine, "007" / "00.5" are codes
            return _FAIL
        try:
            d = Decimal(s)
        except InvalidOperation:
            return _FAIL
        return d if d.is_finite() else _FAIL

    return parse


def _parse_bool(s: str) -> object:
    low = s.lower()
    if low in TRUE_WORDS:
        return True
    if low in FALSE_WORDS:
        return False
    return _FAIL


def _parse_datetimes(uniq: list[str], opts: ParseOptions) -> tuple[dict[str, object], bool]:
    """Parse unique strings in one vectorized call. Returns (mapping, is_tz_aware)."""
    candidates = [u for u in uniq if DATE_SHAPE_RE.search(u)]
    mapping: dict[str, object] = dict.fromkeys(uniq, _FAIL)
    if not candidates:
        return mapping, False
    # dayfirst must only apply to ambiguous d/m/y strings: with it, pandas would also read
    # ISO "2025-10-01" as 10 January. Year-first strings are parsed as ISO 8601.
    iso = [u for u in candidates if ISO_DATE_RE.match(u)]
    other = [u for u in candidates if not ISO_DATE_RE.match(u)]
    tz_aware = False
    for group, kwargs in ((iso, {"format": "ISO8601"}), (other, {"dayfirst": opts.dayfirst})):
        if not group:
            continue
        parsed = _to_datetime(group, **kwargs)
        tz_aware = tz_aware or getattr(parsed.dt, "tz", None) is not None
        for raw, ts in zip(group, parsed, strict=True):
            if not pd.isna(ts):
                mapping[raw] = ts.to_pydatetime()
    return mapping, tz_aware


def _to_datetime(values: list[str], **kwargs) -> pd.Series:
    kwargs.setdefault("format", "mixed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(pd.Series(values), errors="coerce", **kwargs)
        except (ValueError, TypeError):  # mixed UTC offsets
            return pd.to_datetime(pd.Series(values), errors="coerce", utc=True, **kwargs)


# --- driving logic ----------------------------------------------------------------------------


def _apply(values: list[str | None], mapping: dict[str, object]) -> tuple[list, list[str]]:
    out, failed = [], []
    for v in values:
        if v is None:
            out.append(None)
            continue
        r = mapping[v]
        if r is _FAIL:
            failed.append(v)
            out.append(None)
        else:
            out.append(r)
    return out, failed


def _map_unique(values: list[str | None], parse: Callable[[str], object]) -> dict[str, object]:
    return {u: parse(u) for u in {v for v in values if v is not None}}


def _mapping_for(pg_type: str, values: list[str | None], opts: ParseOptions) -> dict[str, object]:
    if pg_type == "int":
        return _map_unique(values, _int_parser(INT32, opts))
    if pg_type == "bigint":
        return _map_unique(values, _int_parser(INT64, opts))
    if type_family(pg_type) == "numeric":
        return _map_unique(values, _numeric_parser(opts))
    if pg_type == "boolean":
        return _map_unique(values, _parse_bool)
    if pg_type in {"date", "timestamp", "timestamptz"}:
        mapping, _ = _parse_datetimes(list({v for v in values if v is not None}), opts)
        if pg_type == "date":
            mapping = {k: (v.date() if v is not _FAIL else v) for k, v in mapping.items()}
        return mapping
    if pg_type == "text":
        return {v: v for v in values if v is not None}
    raise ConversionError(f"no converter for type {pg_type!r}")


def convert(name: str, values: list[str | None], pg_type: str, opts: ParseOptions) -> Column:
    """Convert to a type the yaml asked for. Strict: every non-null value must parse."""
    out, failed = _apply(values, _mapping_for(pg_type, values, opts))
    if failed:
        examples = ", ".join(repr(f) for f in list(dict.fromkeys(failed))[:5])
        raise ConversionError(
            f"column {name!r}: {len(failed)} value(s) are not valid {pg_type} "
            f"(e.g. {examples}); fix the data or change the type in dataset.yaml"
        )
    return Column(name=name, pg_type=pg_type, values=out, inferred=False)


def infer(name: str, values: list[str | None], opts: ParseOptions, threshold: float) -> Column:
    """Pick the narrowest type that fits >= threshold of non-null values."""
    nonnull = [v for v in values if v is not None]
    if not nonnull:
        return Column(name=name, pg_type="text", values=values, inferred=True)
    uniq = list(dict.fromkeys(nonnull))

    def fit(mapping: dict[str, object]) -> float:
        return sum(mapping[v] is not _FAIL for v in nonnull) / len(nonnull)

    candidates: list[tuple[str, Callable[[], dict[str, object]]]] = [
        ("bigint", lambda: _map_unique(values, _int_parser(INT64, opts))),
        ("numeric", lambda: _map_unique(values, _numeric_parser(opts))),
        ("datetime", lambda: _parse_datetimes(uniq, opts)[0]),
        ("boolean", lambda: _map_unique(values, _parse_bool)),
    ]
    for kind, build in candidates:
        mapping = build()
        if fit(mapping) < threshold:
            continue
        pg_type = kind
        if kind == "datetime":
            pg_type = _datetime_kind(mapping)
            if pg_type == "date":
                mapping = {k: (v.date() if v is not _FAIL else v) for k, v in mapping.items()}
        out, failed = _apply(values, mapping)
        return Column(
            name=name,
            pg_type=pg_type,
            values=out,
            inferred=True,
            nulled=len(failed),
            nulled_examples=list(dict.fromkeys(failed))[:5],
        )
    return Column(name=name, pg_type="text", values=values, inferred=True)


def _datetime_kind(mapping: dict[str, object]) -> str:
    parsed = [v for v in mapping.values() if v is not _FAIL]
    if any(getattr(v, "tzinfo", None) is not None for v in parsed):
        return "timestamptz"
    if all(v.hour == v.minute == v.second == v.microsecond == 0 for v in parsed):
        return "date"
    return "timestamp"
