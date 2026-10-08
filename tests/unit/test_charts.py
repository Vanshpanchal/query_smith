"""ChartSpec inference: the model suggests a type, the code picks valid axes."""

from datetime import date
from decimal import Decimal

import pytest

from ask_data.graph.charts import infer_chart

WINS = (["name", "wins"], [["Gujarat Titans", 11], ["Chennai Super Kings", 10], ["MI", 9]])


def test_categories_and_a_measure_make_a_bar():
    assert infer_chart(*WINS) == {"type": "bar", "x": "name", "y": "wins"}


def test_season_is_a_time_axis():
    cols, rows = ["season", "matches"], [[2008, 58], [2009, 57], [2010, 60]]
    assert infer_chart(cols, rows) == {"type": "line", "x": "season", "y": "matches"}


def test_date_values_are_a_time_axis():
    cols, rows = ["d", "revenue"], [[date(2026, 1, 1), Decimal("10.5")], [date(2026, 2, 1), 12]]
    assert infer_chart(cols, rows)["type"] == "line"


def test_per_season_ranking_uses_season_as_x_and_the_number_as_y():
    cols = ["season", "name", "runs"]
    rows = [[2008, "SE Marsh", 616], [2009, "ML Hayden", 572]]
    assert infer_chart(cols, rows) == {"type": "line", "x": "season", "y": "runs"}


@pytest.mark.parametrize(
    ("cols", "rows"),
    [(["name"], [["Sunrisers Hyderabad"]]), (["name", "wins"], [["GT", 11]]), ([], []),
     (["name"], [])],
)  # fmt: skip
def test_single_values_and_empty_results_get_no_chart(cols, rows):
    assert infer_chart(cols, rows) is None


def test_no_measure_or_wide_single_row_is_a_table():
    assert infer_chart(["a", "b"], [["x", "y"], ["z", "w"]])["type"] == "table"
    assert (
        infer_chart(["name", "city", "matches"], [["Wankhede", "Mumbai", 120]])["type"] == "table"
    )


def test_pie_hint_accepted_for_few_non_negative_shares():
    assert infer_chart(*WINS, hint="pie")["type"] == "pie"


def test_pie_hint_rejected_for_a_trend_or_negative_values():
    assert infer_chart(["season", "n"], [[2008, 1], [2009, 2]], hint="pie")["type"] == "line"
    assert infer_chart(["k", "delta"], [["a", -1], ["b", 2]], hint="pie")["type"] == "bar"


def test_none_hint_means_no_chart():
    assert infer_chart(*WINS, hint="none") is None


def test_hint_can_never_point_at_a_missing_column():
    spec = infer_chart(["name", "label"], [["a", "b"], ["c", "d"]], hint="bar")
    assert spec == {"type": "table", "x": None, "y": None}


def test_too_many_bars_become_a_table():
    rows = [[f"player {i}", i] for i in range(40)]
    assert infer_chart(["name", "runs"], rows)["type"] == "table"


def test_bool_is_not_a_measure():
    assert infer_chart(["name", "won"], [["a", True], ["b", False]])["type"] == "table"
