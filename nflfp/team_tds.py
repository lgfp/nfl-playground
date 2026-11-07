"""Module 3 - expected team rushing and passing touchdowns for a week.

Each side is a Poisson regression (log link) on two things known before kickoff:

    log E[TDs] = a + b1 * (Vegas implied team total - 22) / 5  +  b2 * log(module 1 expected yards of that side)

The implied total is  total_line / 2 + (points the team is favoured by) / 2.  The regression is fitted on last season's games plus this season's earlier weeks
(module 1's expected yards for each of those games as they were predicted at the time) and refitted every week; only earlier games are ever used. With fewer than
`min_rows` usable games (the first weeks of the first season) the league average TDs per team-game is used.

Backtest (analysis/team_tds_backtest.py, 2025-26 weeks 4+): the Vegas total is the one predictor that clearly beats the league average (rush ~5%, pass ~11% lower
squared error); a team's own TD history, opponent adjustment and distance-aware xTD each add little or nothing on top (analysis/td_team.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import tds as T

KEYS = ["season", "week", "off_team", "def_team"]
SIDES = ("rush", "pass")


@dataclass(frozen=True)
class TeamTDConfig:
    ridge: float = 1.0       # penalty on the two coefficients
    min_rows: int = 150      # fewer usable training games than this: league average


def implied_totals(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense, defense) with `implied`, the offense's Vegas implied total (nflverse `spread_line` > 0: home favoured)."""
    h = schedule.assign(off_team=schedule.home_team, def_team=schedule.away_team, fav=schedule.spread_line)
    a = schedule.assign(off_team=schedule.away_team, def_team=schedule.home_team, fav=-schedule.spread_line)
    d = pd.concat([h, a], ignore_index=True)
    d["implied"] = d.total_line / 2 + d.fav / 2
    return d[KEYS + ["implied"]]


def _features(d: pd.DataFrame, side: str) -> np.ndarray:
    return np.column_stack([((d.implied - 22.0) / 5.0).to_numpy(float), np.log(d[f"{side}_exp"].clip(lower=5.0)).to_numpy(float)])


def walk_forward(tds: pd.DataFrame, yards: pd.DataFrame, implied: pd.DataFrame, seasons: list[int], cfg: TeamTDConfig = TeamTDConfig(),
                 last_week: int | None = None) -> pd.DataFrame:
    """Expected rush / pass TDs for every fixture of `seasons`, each week predicted from earlier games only.

    tds: tds.team_game_tds() output; yards: module 1 output (needs rush_exp, pass_exp for the fixtures of `seasons` and of the season before the first one);
    implied: implied_totals() for the same seasons. last_week stops the last season after that week.
    """
    X = yards[KEYS + ["rush_exp", "pass_exp"]].merge(implied, on=KEYS, how="left").merge(tds, on=KEYS, how="left")
    out = []
    for season in seasons:
        for week in sorted(X.loc[X.season == season, "week"].unique()):
            if last_week is not None and season == seasons[-1] and week > last_week:
                break
            todo = X[(X.season == season) & (X.week == week)].copy()
            played = X[X.rush_td.notna() & (((X.season == season) & (X.week < week)) | (X.season == season - 1))]
            cur = played[played.season == season]
            base = cur if len(cur) else played
            if base.empty:
                continue
            for side in SIDES:
                todo[f"{side}_td_exp"] = float(base[f"{side}_td"].mean())
                F, y = _features(played, side), played[f"{side}_td"].to_numpy(float)
                ok = np.isfinite(F).all(1)
                if ok.sum() >= cfg.min_rows:
                    b = T.poisson_fit(F[ok], y[ok], cfg.ridge)
                    Ft = _features(todo, side)
                    good = np.isfinite(Ft).all(1)
                    todo.loc[good, f"{side}_td_exp"] = T.poisson_predict(b, Ft[good])
            todo["total_td_exp"] = todo.rush_td_exp + todo.pass_td_exp
            out.append(todo)
    cols = KEYS + ["implied", "rush_td_exp", "pass_td_exp", "total_td_exp", "rush_td", "pass_td"]
    return pd.concat(out, ignore_index=True)[cols]


def project_week(tds: pd.DataFrame, yards: pd.DataFrame, implied: pd.DataFrame, season: int, week: int, cfg: TeamTDConfig = TeamTDConfig()) -> pd.DataFrame:
    """Expected rush and pass TDs (`rush_td_exp`, `pass_td_exp`) for every fixture of one week, from earlier games only."""
    out = walk_forward(tds, yards, implied, [season], cfg, last_week=week)
    return out[out.week == week].reset_index(drop=True)
