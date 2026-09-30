"""datasets/ipl/transform.py on tiny synthetic Cricsheet files (no download needed)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from ask_data.datasets.config import load_dataset_cfg
from ask_data.datasets.loader import prepare_tables, read_tables

IPL_DIR = Path(__file__).resolve().parents[2] / "datasets" / "ipl"


def _load_transform():
    spec = importlib.util.spec_from_file_location("ipl_transform", IPL_DIR / "transform.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transform = _load_transform()

REGISTRY = {"A One": "r1", "B Two": "r2", "C Three": "r3", "D Four": "r4"}


def _ball(batter, bowler, runs=0, extras=None, wicket=None):
    b = {
        "batter": batter,
        "bowler": bowler,
        "non_striker": "B Two" if batter == "A One" else "A One",
        "runs": {"batter": runs, "extras": sum((extras or {}).values()), "total": 0},
    }
    b["runs"]["total"] = b["runs"]["batter"] + b["runs"]["extras"]
    if extras:
        b["extras"] = extras
    if wicket:
        b["wickets"] = [wicket]
    return b


def _match(season, date, venue, city, outcome, event=None, innings=None, pom=("A One",)):
    info = {
        "season": season,
        "dates": [date],
        "venue": venue,
        "event": event or {"name": "Indian Premier League", "match_number": 1},
        "teams": ["Kings XI Punjab", "Delhi Daredevils"],
        "toss": {"winner": "Delhi Daredevils", "decision": "field"},
        "outcome": outcome,
        "players": {
            "Kings XI Punjab": ["A One", "B Two"],
            "Delhi Daredevils": ["C Three", "D Four"],
        },
        "registry": {"people": REGISTRY},
    }
    if city:
        info["city"] = city
    if pom:
        info["player_of_match"] = list(pom)
    return {"meta": {}, "info": info, "innings": innings or []}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    first = _match(
        "2007/08",
        "2008-04-18",
        "Punjab Cricket Association Stadium, Mohali",
        "Chandigarh",
        {"winner": "Kings XI Punjab", "by": {"runs": 12}},
        innings=[
            {
                "team": "Kings XI Punjab",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            _ball("A One", "C Three", 4),
                            _ball("A One", "C Three", extras={"noballs": 1, "byes": 4}),
                            _ball("A One", "C Three", extras={"wides": 1}),
                            _ball(
                                "A One",
                                "C Three",
                                wicket={
                                    "kind": "caught",
                                    "player_out": "A One",
                                    "fielders": [{"name": "D Four"}],
                                },
                            ),
                        ],
                    }
                ],
            }
        ],
    )
    tie = _match(
        "2020/21",
        "2020-09-20",
        "Dubai International Cricket Stadium",
        None,
        {"result": "tie", "eliminator": "Delhi Daredevils"},
        event={"name": "Indian Premier League", "stage": "Final"},
    )
    washed_out = _match(
        2021, "2021-05-01", "Feroz Shah Kotla", "Delhi", {"result": "no result"}, pom=()
    )
    for mid, doc in [(100, first), (200, tie), (300, washed_out)]:
        (tmp_path / f"{mid}.json").write_text(json.dumps(doc))
    return tmp_path


@pytest.fixture
def frames(raw_dir):
    return transform.extract(raw_dir)


@pytest.mark.parametrize(
    ("season", "date", "expected"),
    [
        ("2007/08", "2008-04-18", 2008),
        ("2009/10", "2010-03-12", 2010),
        ("2020/21", "2020-09-19", 2020),
        (2017, "2017-04-05", 2017),
    ],
)
def test_normalize_season(season, date, expected):
    assert transform.normalize_season(season, date) == expected


def test_unexpected_season_format_fails_loudly():
    with pytest.raises(ValueError, match="season"):
        transform.normalize_season("IPL 2008", "2008-04-18")


def test_matches_row_shapes(frames):
    m = frames["matches"].set_index("id")
    assert m.loc[100, "season"] == 2008
    assert (m.loc[100, "result"], m.loc[100, "win_margin"]) == ("runs", 12)
    assert m.loc[200, "season"] == 2020
    assert m.loc[200, "stage"] == "Final"
    assert m.loc[200, "result"] == "tie"
    assert m.loc[200, "winner_id"] == m.loc[200, "toss_winner_id"]  # super-over winner
    assert m.loc[300, "result"] == "no result"
    assert m[["winner_id", "player_of_match_id"]].loc[300].isna().all()


def test_renamed_venues_merge_and_city_is_filled(frames):
    venues = frames["venues"].set_index("name")["city"].to_dict()
    assert venues == {
        "Punjab Cricket Association IS Bindra Stadium": "Chandigarh",
        "Dubai International Cricket Stadium": "Dubai",
        "Arun Jaitley Stadium": "Delhi",
    }


def test_renamed_teams_stay_separate_but_share_a_franchise(frames):
    teams = frames["teams"].set_index("name")["franchise"].to_dict()
    assert teams == {"Kings XI Punjab": "Punjab Kings", "Delhi Daredevils": "Delhi Capitals"}


def test_ids_follow_first_appearance(frames):
    assert frames["teams"]["id"].tolist() == [1, 2]
    assert frames["players"]["registry_id"].tolist()[:1] == ["r1"]


def test_deliveries(frames):
    d = frames["deliveries"]
    assert d["ball_no"].tolist() == [1, 2, 3, 4]
    # no-ball + byes on one ball: the no-ball decides legality
    extras = [None if pd.isna(e) else e for e in d["extras_type"]]
    assert extras == [None, "noballs", "wides", None]
    wicket = d.iloc[3]
    assert bool(wicket["is_wicket"]) and wicket["wicket_kind"] == "caught"
    players = frames["players"].set_index("id")["name"]
    assert players[wicket["fielder_id"]] == "D Four"
    assert players[wicket["player_out_id"]] == "A One"
    assert set(d["bowling_team_id"]) == {2}


def test_squads(frames):
    assert len(frames["player_match"]) == 3 * 4


def test_empty_raw_dir_explains_how_to_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_ipl"):
        transform.extract(tmp_path)


def test_output_passes_the_generic_loader_checks(raw_dir, tmp_path, monkeypatch):
    """The real dataset.yaml + transform.py produce tables the loader accepts (PK/FK/types)."""
    cfg = load_dataset_cfg(IPL_DIR)
    monkeypatch.setattr(cfg.source, "path", str(raw_dir))  # absolute path wins in `/`
    tables = prepare_tables(cfg, read_tables(cfg, IPL_DIR))
    assert tables["deliveries"].row_count == 4
    assert tables["matches"].column("season").pg_type == "int"
