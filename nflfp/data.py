"""Download (and cache) nflverse weekly player stats and the schedule."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(".cache")
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def _fetch(url: str, name: str, refresh: bool) -> Path:
    path = CACHE / name
    if refresh or not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_player_weeks(season: int, refresh: bool = False) -> pd.DataFrame:
    pw = pd.read_csv(_fetch(STATS_URL.format(season=season), f"stats_{season}.csv", refresh), low_memory=False)
    return pw[pw["season_type"] == "REG"]


def load_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    g = pd.read_csv(_fetch(GAMES_URL, "games.csv", refresh), low_memory=False)
    return g[(g["season"] == season) & (g["game_type"] == "REG")]


def next_matchups(schedule: pd.DataFrame, week: int | None = None) -> tuple[int, list[tuple[str, str]]]:
    """(week, [(offense, defense), ...]) - both directions for each game projected.

    Default: the earliest week with an unplayed game, and only its unplayed games.
    An explicit week returns every game in that week, played or not.
    """
    if week is None:
        unplayed = schedule[schedule["result"].isna()]
        if unplayed.empty:
            raise SystemExit("No unplayed games left in this season; pass --week explicitly.")
        week = int(unplayed["week"].min())
        games = unplayed[unplayed["week"] == week]
    else:
        games = schedule[schedule["week"] == week]
    pairs = [(r.away_team, r.home_team) for r in games.itertuples()]
    pairs += [(h, a) for a, h in pairs]
    return week, pairs
