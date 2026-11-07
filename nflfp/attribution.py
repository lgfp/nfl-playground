"""Module 2 - player attribution: split module 1's team yards among the players using Sleeper's projected volume.

Module 1 gives each offense's expected rush and pass yards. Here they become fantasy points per player:

    rushing yards   = team rush yards   x  player's share of the team's projected carries   (Sleeper `rush_att`)
    receiving yards = team pass yards   x  player's share of the team's projected targets   (Sleeper `rec_tgt`)
    passing yards   = team pass yards   x  player's share of the team's projected attempts  (Sleeper `pass_att`; in practice the starter)

Touchdowns are left out (too noisy to project); the points are yards plus receptions only. Receptions follow the yards at the
league's rate this season so far (receptions per pass yard), shared out by the same target shares.
Players Sleeper lists as Out / IR are removed first, so their volume is spread over their teammates in proportion to
the teammates' own projected volume.

Sleeper's feed is `api.sleeper.app/projections/nfl/{season}/{week}` (public, no key). Only the volume is used, none of Sleeper's
yards, TDs or points.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

from . import sleeper
from .model import Scoring

URL = "https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE"
VOLUME = {"rush_att": "carries", "rec_tgt": "targets", "pass_att": "attempts"}


@dataclass(frozen=True)
class Rates:
    """League conversion of team yards into receptions."""
    rec_per_pass_yd: float


# ------------------------------------------------------------------------------------------------------ inputs
def league_rates(player_weeks: pd.DataFrame) -> Rates:
    """Rates from nflverse player-weeks (pass in only the games before the week being projected)."""
    s = player_weeks[["passing_yards", "receptions"]].sum()
    return Rates(rec_per_pass_yd=float(s.receptions / s.passing_yards))


def sleeper_volume(season: int, week: int) -> pd.DataFrame:
    """Projected volume of every QB/RB/WR/TE Sleeper projects for the week: sleeper_id, name, position, team (nflverse code), injury_status,
    carries, targets, attempts."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL.format(season=season, week=week), timeout=60) as r:
                raw = json.loads(r.read().decode())
            break
        except (urllib.error.URLError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    rows = []
    for x in raw:
        st, pl = x.get("stats") or {}, x.get("player") or {}
        team = sleeper.TEAM.get(x.get("team"), x.get("team"))
        if not team or not any(k in st for k in VOLUME):
            continue
        rows.append(dict(sleeper_id=str(x["player_id"]), name=f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip(),
                         position=pl.get("position"), team=team, injury_status=pl.get("injury_status"),
                         **{v: float(st.get(k, 0.0)) for k, v in VOLUME.items()}))
    return pd.DataFrame(rows, columns=["sleeper_id", "name", "position", "team", "injury_status", *VOLUME.values()]).drop_duplicates("sleeper_id")


def shares(volume: pd.DataFrame, exclude: set | frozenset = frozenset(), out_statuses=sleeper.OUT_STATUSES) -> pd.DataFrame:
    """Each player's share of his team's projected carries / targets / attempts, after removing `exclude` (Sleeper ids) and Out/IR players.
    Shares of a team sum to 1 (0 for a team with no volume of that kind)."""
    v = volume[~volume.sleeper_id.isin(exclude) & ~volume.injury_status.isin(out_statuses)].copy()
    for col in VOLUME.values():
        tot = v.groupby("team")[col].transform("sum")
        v[f"{col}_share"] = (v[col] / tot).where(tot > 0, 0.0)
    return v


# ---------------------------------------------------------------------------------------------------- the model
def attribute(team_yards: pd.DataFrame, sh: pd.DataFrame, rates: Rates, scoring: Scoring = Scoring()) -> pd.DataFrame:
    """Expected fantasy points per player.

    team_yards: module 1 rows (needs off_team, def_team, rush_exp, pass_exp); sh: shares() output.
    Returns one row per player with volume on a team in team_yards: expected rush/rec/pass yards, TDs, receptions and points.
    """
    t = team_yards[["off_team", "def_team", "rush_exp", "pass_exp"]].rename(columns={"off_team": "team", "def_team": "opp"})
    d = sh.merge(t, on="team", how="inner")
    d["rush_yds"] = d.rush_exp * d.carries_share
    d["rec_yds"] = d.pass_exp * d.targets_share
    d["receptions"] = d.rec_yds * rates.rec_per_pass_yd
    d["pass_yds"] = d.pass_exp * d.attempts_share
    d["rush_pts"] = scoring.yard * d.rush_yds
    d["rec_pts"] = scoring.yard * d.rec_yds + scoring.ppr * d.receptions
    d["pass_pts"] = scoring.pass_yard * d.pass_yds
    d["exp_pts"] = d.rush_pts + d.rec_pts + d.pass_pts
    cols = ["sleeper_id", "name", "position", "team", "opp", "carries", "targets", "attempts", "carries_share", "targets_share", "attempts_share",
            "rush_yds", "rec_yds", "pass_yds", "receptions", "rush_pts", "rec_pts", "pass_pts", "exp_pts"]
    return d[cols].sort_values("exp_pts", ascending=False).reset_index(drop=True)
