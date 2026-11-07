"""Download (and cache) nflverse weekly player stats and the schedule."""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(".cache")
CACHE_TTL_HOURS = 24  # nflverse updates nightly (more often only on game days); a finished season never changes
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
PBP_COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type", "two_point_attempt", "rushing_yards", "passing_yards"]
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def current_season(today: pd.Timestamp | None = None) -> int:
    """The NFL season in progress: it starts in September and the new league year starts in March."""
    t = today if today is not None else pd.Timestamp.now()
    return t.year if t.month >= 3 else t.year - 1


def _fetch(url: str, name: str, refresh: bool, ttl_hours: float | None = None) -> Path:
    """Download `url` to the cache unless a fresh copy exists (`ttl_hours=None`: a cached copy never expires).

    The download goes to a temporary file first, so an interrupted transfer cannot leave a broken cache. If a refresh
    fails but an older copy exists, that copy is used (with a warning) instead of failing.
    """
    path = CACHE / name
    age_h = (time.time() - path.stat().st_mtime) / 3600 if path.exists() else None
    if refresh or age_h is None or (ttl_hours is not None and age_h > ttl_hours):
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if age_h is None:
                raise
            print(f"warning: could not refresh {name} ({e}); using the copy downloaded {age_h:.0f}h ago", file=sys.stderr)
    return path


def load_player_weeks(season: int, refresh: bool = False) -> pd.DataFrame:
    ttl = None if season < current_season() else CACHE_TTL_HOURS  # past seasons are final
    pw = pd.read_csv(_fetch(STATS_URL.format(season=season), f"stats_{season}.csv", refresh, ttl), low_memory=False)
    return pw[pw["season_type"] == "REG"]


def load_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    g = pd.read_csv(_fetch(GAMES_URL, "games.csv", refresh, CACHE_TTL_HOURS), low_memory=False)
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


def load_pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    """Regular-season run and pass plays (no 2-point tries), with just the columns the play-level model needs."""
    ttl = None if season < current_season() else CACHE_TTL_HOURS
    d = pd.read_csv(_fetch(PBP_URL.format(season=season), f"pbp_{season}.csv.gz", refresh, ttl), usecols=PBP_COLS, low_memory=False)
    d = d[(d["season_type"] == "REG") & d["posteam"].notna() & d["play_type"].isin(["run", "pass"]) & (d["two_point_attempt"] != 1)]
    return d.drop(columns=["season_type", "two_point_attempt"]).reset_index(drop=True)
