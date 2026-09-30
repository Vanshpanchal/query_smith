"""Cricsheet IPL JSON (one file per match) -> the 6 tables declared in dataset.yaml.

Called by the generic loader (source.type: python) as extract(raw_dir). This file is the
ONLY place IPL-specific parsing lives; src/ask_data/ never knows about cricket.

Normalization done here (see notes.md for the rules the agent relies on):
  * season  : "2007/08" -> 2008, "2009/10" -> 2010, "2020/21" -> 2020 (year of the first match)
  * venues  : "Eden Gardens, Kolkata" and "Eden Gardens" are one ground; renamed grounds
              (Feroz Shah Kotla -> Arun Jaitley Stadium, ...) are merged into the current name
  * teams   : renamed teams stay separate rows (historically accurate); `franchise` groups them
  * ids     : teams/venues/players get integer ids in order of first appearance, so adding a
              new season appends ids instead of renumbering existing ones
  * ties    : winner_id = super-over winner (outcome.eliminator); super-over balls are inning 3+
"""

import json
import re
from pathlib import Path

import pandas as pd

# Same ground, different spellings or an official rename -> one canonical name
VENUE_ALIASES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Zayed Cricket Stadium": "Sheikh Zayed Stadium",
}
CITY_ALIASES = {"Bangalore": "Bengaluru"}
# Venues whose files never name a city (the UAE 2020 season)
VENUE_CITY_FALLBACK = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
}

# Renamed teams -> current franchise name. Teams not listed are their own franchise.
FRANCHISES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# A ball can carry two extras (no-ball + byes); the one that decides ball legality wins
EXTRAS_PRIORITY = ["wides", "noballs", "penalty", "legbyes", "byes"]

SEASON_RE = re.compile(r"^\d{4}(/\d{2})?$")


class _Ids:
    """name/key -> integer id, assigned in order of first appearance."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def __call__(self, key: str | None) -> int | None:
        if key is None:
            return None
        if key not in self.ids:
            self.ids[key] = len(self.ids) + 1
        return self.ids[key]


def extract(raw_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(Path(raw_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no Cricsheet *.json files in {raw_dir}; run scripts/fetch_ipl.py")
    docs = [(int(p.stem), json.loads(p.read_text())) for p in files]
    # Chronological, so ids and "latest name/city" are stable and meaningful
    docs.sort(key=lambda d: (d[1]["info"]["dates"][0], d[0]))

    team_ids, venue_ids, player_ids = _Ids(), _Ids(), _Ids()
    venue_city: dict[str, str] = {}
    player_name: dict[str, str] = {}  # registry id -> latest name used
    matches, squads, deliveries = [], [], []

    for match_id, doc in docs:
        info = doc["info"]
        registry: dict[str, str] = info["registry"]["people"]

        def pid(name: str | None, registry=registry) -> int | None:
            if name is None:
                return None
            reg_id = registry[name]
            player_name[reg_id] = name
            return player_ids(reg_id)

        for team in info["teams"]:
            team_ids(team)
        venue = _venue_name(info["venue"])
        city = _city(info.get("city")) or VENUE_CITY_FALLBACK.get(venue)
        if city:
            venue_city[venue] = city

        matches.append(_match_row(match_id, info, team_ids, venue_ids(venue), pid))
        for team, names in info["players"].items():
            for name in dict.fromkeys(names):
                squads.append(
                    {"match_id": match_id, "player_id": pid(name), "team_id": team_ids(team)}
                )
        deliveries += _delivery_rows(match_id, doc["innings"], info["teams"], team_ids, pid)

    teams = pd.DataFrame(
        {"id": i, "name": name, "franchise": FRANCHISES.get(name, name)}
        for name, i in team_ids.ids.items()
    )
    venues = pd.DataFrame(
        {"id": i, "name": name, "city": venue_city.get(name)} for name, i in venue_ids.ids.items()
    )
    players = pd.DataFrame(
        {"id": i, "registry_id": reg_id, "name": player_name[reg_id]}
        for reg_id, i in player_ids.ids.items()
    )
    return {
        "teams": teams,
        "venues": venues,
        "players": players,
        "matches": _int_columns(pd.DataFrame(matches)),
        "player_match": pd.DataFrame(squads),
        "deliveries": _int_columns(pd.DataFrame(deliveries)),
    }


def _match_row(match_id: int, info: dict, team_ids: _Ids, venue_id: int, pid) -> dict:
    event = info.get("event", {})
    outcome = info["outcome"]
    toss = info.get("toss", {})
    team1, team2 = info["teams"]

    if "winner" in outcome:
        ((result, margin),) = outcome["by"].items()  # {"runs": 35} or {"wickets": 5}
        winner = outcome["winner"]
    else:  # {"result": "tie", "eliminator": "..."} or {"result": "no result"}
        result, margin = outcome["result"], None
        winner = outcome.get("eliminator")

    potm = info.get("player_of_match") or [None]
    return {
        "id": match_id,
        "season": normalize_season(info["season"], info["dates"][0]),
        "match_date": info["dates"][0],
        "match_number": event.get("match_number"),
        "stage": event.get("stage"),
        "venue_id": venue_id,
        "team1_id": team_ids(team1),
        "team2_id": team_ids(team2),
        "toss_winner_id": team_ids(toss.get("winner")),
        "toss_decision": toss.get("decision"),
        "winner_id": team_ids(winner),
        "result": result,
        "win_margin": margin,
        "method": outcome.get("method"),
        "player_of_match_id": pid(potm[0]),
    }


def _delivery_rows(match_id: int, innings: list, teams: list[str], team_ids: _Ids, pid) -> list:
    rows = []
    for inning_no, inning in enumerate(innings, start=1):
        batting = inning["team"]
        bowling = teams[1] if batting == teams[0] else teams[0]
        for over in inning["overs"]:
            for ball_no, ball in enumerate(over["deliveries"], start=1):
                extras = ball.get("extras", {})
                wicket = (ball.get("wickets") or [None])[0]
                fielders = (wicket or {}).get("fielders") or [{}]
                rows.append(
                    {
                        "match_id": match_id,
                        "inning": inning_no,
                        "over_no": over["over"],
                        "ball_no": ball_no,
                        "batting_team_id": team_ids(batting),
                        "bowling_team_id": team_ids(bowling),
                        "batter_id": pid(ball["batter"]),
                        "bowler_id": pid(ball["bowler"]),
                        "non_striker_id": pid(ball["non_striker"]),
                        "runs_batter": ball["runs"]["batter"],
                        "runs_extras": ball["runs"]["extras"],
                        "runs_total": ball["runs"]["total"],
                        "extras_type": next((e for e in EXTRAS_PRIORITY if e in extras), None),
                        "is_wicket": wicket is not None,
                        "wicket_kind": wicket["kind"] if wicket else None,
                        "player_out_id": pid(wicket["player_out"]) if wicket else None,
                        "fielder_id": pid(fielders[0].get("name")),
                    }
                )
    return rows


def normalize_season(season: str | int, first_date: str) -> int:
    """IPL edition year: '2007/08' -> 2008 (the year the matches were played)."""
    s = str(season)
    if not SEASON_RE.match(s):
        raise ValueError(f"unexpected season format: {season!r}")
    return int(first_date[:4]) if "/" in s else int(s)


def _venue_name(raw: str) -> str:
    name = raw.split(",")[0].strip()  # drop ", Kolkata" / ", Uppal, Hyderabad" suffixes
    return VENUE_ALIASES.get(name, name)


def _city(raw: str | None) -> str | None:
    return CITY_ALIASES.get(raw, raw) if raw else None


def _int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable int columns (ids with some None) as Int64, not float, so no 1.0 ids appear."""
    for col in df.columns:
        if col.endswith("_id") or col in {"match_number", "win_margin"}:
            df[col] = df[col].astype("Int64")
    return df
