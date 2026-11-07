"""Shared pieces for the touchdown modules (3 and 4): play-by-play with an expected-TD value per play, team-game TD counts, a Poisson fit.

xTD of a play = the league's scoring rate for that play type and spot on the field (yard-line bin), taken from the PREVIOUS season's plays so a
season never explains itself (the first season loaded uses its own plays). Only offensive rushing / passing TDs count (no returns, no 2-point tries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

TD_COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type", "two_point_attempt", "qb_kneel", "qb_spike", "yardline_100",
           "rush_touchdown", "pass_touchdown", "rusher_player_id", "rusher_player_name", "receiver_player_id", "receiver_player_name"]
EDGES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]      # yard-line bins (yards from the opponent's end zone)


def load_plays(seasons, refresh: bool = False) -> pd.DataFrame:
    """Regular-season run and pass plays (no 2-point tries, kneels or spikes) for the seasons, with `td` (offensive TD on the play) and `bin`."""
    parts = []
    for y in seasons:
        ttl = None if y < data.current_season() else data.CACHE_TTL_HOURS
        parts.append(pd.read_csv(data._fetch(data.PBP_URL.format(season=y), f"pbp_{y}.csv.gz", refresh, ttl), usecols=TD_COLS, low_memory=False))
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.season_type == "REG") & d.posteam.notna() & d.play_type.isin(["run", "pass"]) & (d.two_point_attempt != 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    d["td"] = np.where(d.play_type == "run", d.rush_touchdown, d.pass_touchdown).astype(float)
    d["bin"] = pd.cut(d.yardline_100, EDGES, labels=False).astype(float)
    return d.drop(columns=["season_type", "two_point_attempt", "qb_kneel", "qb_spike"]).reset_index(drop=True)


def with_xtd(plays: pd.DataFrame) -> pd.DataFrame:
    """Adds `xtd` (see the module docstring)."""
    out = []
    for s, p in plays.groupby("season"):
        ref = plays[plays.season == s - 1] if (plays.season == s - 1).any() else p
        out.append(p.join(ref.groupby(["play_type", "bin"]).td.mean().rename("xtd"), on=["play_type", "bin"]))
    r = pd.concat(out).sort_index()
    r["xtd"] = r.xtd.fillna(r.groupby("play_type").td.transform("mean"))
    return r


def team_game_tds(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, off_team, def_team): rush_td and pass_td scored by the offense."""
    g = plays.assign(rush_td=np.where(plays.play_type == "run", plays.td, 0.0), pass_td=np.where(plays.play_type == "pass", plays.td, 0.0))
    t = g.groupby(["season", "week", "posteam", "defteam"], as_index=False)[["rush_td", "pass_td"]].sum()
    return t.rename(columns={"posteam": "off_team", "defteam": "def_team"})


def poisson_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 30) -> np.ndarray:
    """Ridge-penalised Poisson regression with a log link, by IRLS. X has no intercept column; returns [intercept, coefficients] (the intercept is not penalised)."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1])
    b[0] = np.log(max(float(y.mean()), 1e-3))
    P = ridge * np.eye(A.shape[1])
    P[0, 0] = 0.0
    for _ in range(iters):
        mu = np.exp(np.clip(A @ b, -10, 10))
        nb = np.linalg.solve(A.T @ (A * mu[:, None]) + P, A.T @ (mu * (A @ b + (y - mu) / mu)))
        done = np.max(np.abs(nb - b)) < 1e-8
        b = nb
        if done:
            break
    return b


def poisson_predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(b[0] + X @ b[1:], -10, 10))
