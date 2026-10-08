"""ChartSpec for a result: which chart, and which columns are its axes.

The answer model only suggests the chart *type* (it knows "share" means a pie and "per season"
a trend); this code picks the columns and checks the data fits, so a bad suggestion can never
produce a chart that points at a missing or non-numeric column.
"""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ask_data.graph.state import ChartSpec

TIME_NAME_RE = re.compile(r"(^|_)(season|year|month|quarter|week|date|day)s?($|_)")
MAX_BAR_ROWS = 30
MAX_PIE_ROWS = 8


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float | Decimal) and not isinstance(v, bool)


def _numeric_columns(columns: list[str], rows: list[list[Any]]) -> list[int]:
    out = []
    for i in range(len(columns)):
        values = [r[i] for r in rows if r[i] is not None]
        if values and all(_is_number(v) for v in values):
            out.append(i)
    return out


def _is_time(name: str, rows: list[list[Any]], i: int) -> bool:
    values = [r[i] for r in rows if r[i] is not None]
    if values and all(isinstance(v, date | datetime) for v in values):
        return True
    return bool(TIME_NAME_RE.search(name.lower()))


def infer_chart(
    columns: list[str], rows: list[list[Any]], hint: str | None = None
) -> ChartSpec | None:
    """None when a chart adds nothing (no rows, one value); `table` when the shape has no
    obvious axes; otherwise bar/line/pie with x = label (or time) column, y = first measure."""
    if not rows or not columns or hint == "none":
        return None
    if len(rows) == 1:
        return None if len(columns) <= 2 else {"type": "table", "x": None, "y": None}

    numeric = _numeric_columns(columns, rows)
    time_cols = [i for i in range(len(columns)) if _is_time(columns[i], rows, i)]
    labels = [i for i in range(len(columns)) if i not in numeric]
    # x: a time column if there is one (a trend), else the first text column (categories)
    x = time_cols[0] if time_cols else (labels[0] if labels else None)
    measures = [i for i in numeric if i != x and i not in time_cols]
    if x is None or not measures:
        return {"type": "table", "x": None, "y": None}
    y = measures[0]
    spec: ChartSpec = {"type": "line" if time_cols else "bar", "x": columns[x], "y": columns[y]}

    ys = [r[y] for r in rows if r[y] is not None]
    if hint == "pie" and len(rows) <= MAX_PIE_ROWS and all(v >= 0 for v in ys) and not time_cols:
        spec["type"] = "pie"
    elif hint in ("bar", "line"):
        spec["type"] = hint  # both fit any x/y shape; the model may know better (ordered ranks)
    if spec["type"] == "bar" and len(rows) > MAX_BAR_ROWS:
        return {"type": "table", "x": None, "y": None}
    return spec
