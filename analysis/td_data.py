"""Shared data layer for the touchdown experiments: plays with an expected-TD value, team-game and player-game tables, game environment, Poisson GLM.

Only offensive rushing / passing TDs are counted (no returns, no 2-point tries). Seasons used: 2024, 2025, 2026 (an experiment never looks further back than the previous season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type", "two_point_attempt", "qb_kneel", "qb_spike", "yardline_100", "goal_to_go",
        "rush_touchdown", "pass_touchdown", "rusher_player_id", "rusher_player_name", "receiver_player_id", "receiver_player_name",
        "passer_player_id", "rushing_yards", "passing_yards", "qb_scramble", "sack", "complete_pass"]
EDGES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
SEASONS = (2024, 2025, 2026)


def load_plays(seasons=SEASONS) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f".cache/pbp_{y}.csv.gz", usecols=COLS, low_memory=False) for y in seasons], ignore_index=True)
    d = d[(d.season_type == "REG") & d.posteam.notna() & d.play_type.isin(["run", "pass"]) & (d.two_point_attempt != 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    d["td"] = np.where(d.play_type == "run", d.rush_touchdown, d.pass_touchdown).astype(float)
    d["bin"] = pd.cut(d.yardline_100, EDGES, labels=False).astype(float)
    d["rushing_yards"] = d.rushing_yards.fillna(0.0)
    d["passing_yards"] = d.passing_yards.fillna(0.0)
    return d.reset_index(drop=True)


def curve(plays: pd.DataFrame) -> pd.Series:
    """League scoring rate by (play type, yard-line bin)."""
    return plays.groupby(["play_type", "bin"]).td.mean()


def with_xtd(plays: pd.DataFrame) -> pd.DataFrame:
    """Adds `xtd`: the scoring rate at that spot from the PREVIOUS season's curve (the season itself for the first one)."""
    out = []
    for s, p in plays.groupby("season"):
        ref = plays[plays.season == s - 1] if (plays.season == s - 1).any() else p
        c = curve(ref).rename("xtd")
        out.append(p.join(c, on=["play_type", "bin"]))
    r = pd.concat(out).sort_index()
    r["xtd"] = r.xtd.fillna(r.groupby("play_type").td.transform("mean"))
    return r


def team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team): opponent, rush/pass TDs, xTDs, yards, play counts."""
    g = plays.assign(rtd=np.where(plays.play_type == "run", plays.td, 0.0), ptd=np.where(plays.play_type == "pass", plays.td, 0.0),
                     rx=np.where(plays.play_type == "run", plays.xtd, 0.0), px=np.where(plays.play_type == "pass", plays.xtd, 0.0),
                     rn=(plays.play_type == "run").astype(float), pn=(plays.play_type == "pass").astype(float))
    t = g.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        rush_td=("rtd", "sum"), pass_td=("ptd", "sum"), rush_x=("rx", "sum"), pass_x=("px", "sum"),
        rush_plays=("rn", "sum"), pass_plays=("pn", "sum"), rush_yds=("rushing_yards", "sum"), pass_yds=("passing_yards", "sum"))
    return t.rename(columns={"posteam": "team", "defteam": "opp"})


def environment(seasons=SEASONS) -> pd.DataFrame:
    """Per (season, week, team): points the team is favoured by and its implied total (total/2 + spread/2)."""
    g = pd.read_csv(".cache/games.csv", low_memory=False)
    g = g[g.season.isin(seasons) & (g.game_type == "REG")]
    h = g.assign(team=g.home_team, fav=g.spread_line)
    a = g.assign(team=g.away_team, fav=-g.spread_line)
    e = pd.concat([h, a])[["season", "week", "team", "fav", "total_line"]]
    e["implied"] = e.total_line / 2 + e.fav / 2
    return e


# ------------------------------------------------------------------------------------------------ Poisson GLM
def poisson_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 30) -> np.ndarray:
    """Ridge-penalised Poisson regression, log link, IRLS. X has no intercept column; returns [intercept, coefs]. The intercept is not penalised."""
    A = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(A.shape[1]); b[0] = np.log(max(y.mean(), 1e-3))
    P = lam * np.eye(A.shape[1]); P[0, 0] = 0.0
    for _ in range(iters):
        mu = np.exp(np.clip(A @ b, -10, 10))
        W = mu
        z = A @ b + (y - mu) / mu
        nb = np.linalg.solve(A.T @ (A * W[:, None]) + P, A.T @ (W * z))
        if np.max(np.abs(nb - b)) < 1e-8:
            b = nb
            break
        b = nb
    return b


def poisson_predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(b[0] + X @ b[1:], -10, 10))


def ridge_mean(total, n, k, target):
    """Mean with k pseudo-observations of `target`."""
    return (total + k * target) / (n + k)


def poisson_deviance(y, mu) -> float:
    mu = np.maximum(mu, 1e-6)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))
