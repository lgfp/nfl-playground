"""Rush chain = ridge -> Vegas (free slope). How much does the ridge setting matter?

ridge:  rush estimate = (n * team average + K * target) / (n + K),  target = w * last season's team average + (1 - w) * league average
Vegas:  + (yards per point favoured, fitted on last season + this season's earlier weeks, on the ridge's residuals) * spread

Grid over K (pseudo-games) and w (weight on last season), walk-forward on 2025-2026.

    uv run python -m analysis.rush_ridge
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from analysis.ablation import load_all
from analysis.yards_backtest import MAIN_FROM

KS = (2, 4, 8, 12, 16, 24, 32, 48, 64)
WS = (0.0, 0.15, 0.3, 0.5, 0.75)


def features(G: pd.DataFrame) -> pd.DataFrame:
    """For every team-game: games played so far (n), team average so far, last season's team average, league average so far."""
    rows = []
    for season in sorted(G.season.unique()):
        prev = G[G.season == season - 1]
        prev_avg = prev.groupby("off_team").rush_yds.mean() if len(prev) else pd.Series(dtype=float)
        prev_lg = prev.rush_yds.mean() if len(prev) else np.nan
        for week in sorted(G.loc[G.season == season, "week"].unique()):
            cur, todo = G[(G.season == season) & (G.week < week)], G[(G.season == season) & (G.week == week)].copy()
            todo["n"] = todo.off_team.map(cur.groupby("off_team").size()).fillna(0.0)
            todo["own"] = todo.off_team.map(cur.groupby("off_team").rush_yds.mean())
            todo["pa"] = todo.off_team.map(prev_avg)
            todo["L"] = cur.rush_yds.mean() if len(cur) else prev_lg
            rows.append(todo)
    return pd.concat(rows, ignore_index=True)


def chain(F: pd.DataFrame, K: float, w: float, vegas: bool = True) -> np.ndarray:
    L = F.L.to_numpy(float)
    own = np.where(F.n > 0, F.own, L)
    pa = np.where(F.pa.notna(), F.pa, L)
    base = (F.n.to_numpy(float) * own + K * (w * pa + (1 - w) * L)) / (F.n.to_numpy(float) + K)
    if not vegas:
        return base
    pred = base.copy()
    res, fav = F.rush_yds.to_numpy(float) - base, F.fav.to_numpy(float)
    season, week = F.season.to_numpy(), F.week.to_numpy()
    for s in np.unique(season):
        for wk in np.unique(week[season == s]):
            tr = ((season == s - 1) | ((season == s) & (week < wk))) & np.isfinite(res)
            now = (season == s) & (week == wk)
            den = float((fav[tr] ** 2).sum())
            slope = float((fav[tr] * res[tr]).sum() / den) if den else 0.0
            pred[now] = np.maximum(base[now] + slope * fav[now], 0.0)
    return pred


def score(F: pd.DataFrame, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    a, p = F.rush_yds.to_numpy(float)[mask], pred[mask]
    ok = np.isfinite(p)
    a, p = a[ok], p[ok]
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum(), np.abs(a - p).mean(), np.corrcoef(a, p)[0, 1]


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    F = features(load_all())
    ev = F.season.isin([2025, 2026]).to_numpy()
    late, early = ev & (F.week >= MAIN_FROM).to_numpy(), ev & (F.week < MAIN_FROM).to_numpy()
    res = {}
    for K, w in itertools.product(KS, WS):
        p_v, p_r = chain(F, K, w), chain(F, K, w, vegas=False)
        res[(K, w)] = dict(R2=score(F, p_v, late)[0], MAE=score(F, p_v, late)[1], R2_ridge_only=score(F, p_r, late)[0], R2_early=score(F, p_v, early)[0])
    g = pd.DataFrame(res).T
    for col, title in (("R2", "rush R2, ridge -> Vegas, weeks 4+"), ("R2_ridge_only", "rush R2, ridge only (no Vegas), weeks 4+"), ("MAE", "rush MAE, ridge -> Vegas, weeks 4+"), ("R2_early", "rush R2, ridge -> Vegas, weeks 1-3")):
        print(f"=== {title}   [rows: ridge strength K (pseudo-games); columns: weight on last season]")
        print(g[col].unstack().to_string()); print()
    best = g.R2.idxmax(); default = (8, 0.3)
    a = F.rush_yds.to_numpy(float)[late]
    pb, pd_ = chain(F, *best)[late], chain(F, *default)[late]
    d = (a - pd_) ** 2 - (a - pb) ** 2
    idx = np.random.default_rng(0).integers(0, len(d), (3000, len(d)))
    boots = d[idx].mean(1) / ((a - a.mean()) ** 2).mean()
    print(f"best: K={best[0]}, last-season weight {best[1]} -> R2 {g.loc[best,'R2']:.3f}, MAE {g.loc[best,'MAE']:.2f}; default (K=8, w=0.3): R2 {g.loc[default,'R2']:.3f}, MAE {g.loc[default,'MAE']:.2f}")
    print(f"change best vs default: dR2 {d.mean() / ((a - a.mean()) ** 2).mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")
    for s in (2025, 2026):
        m = (F.season == s).to_numpy() & (F.week >= MAIN_FROM).to_numpy()
        print(f"  {s}: best R2 {score(F, chain(F, *best), m)[0]:.3f} | default {score(F, chain(F, *default), m)[0]:.3f} | no ridge change K=16 w=0.3 {score(F, chain(F, 16, 0.3), m)[0]:.3f}")
