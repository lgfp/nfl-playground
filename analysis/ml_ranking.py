"""Ranking quality: do projections put players in the right ORDER within a position and week? (Start/sit only needs the order.)

Per position-week, over the players in the pool:
  spearman     rank correlation of projected and actual points
  pairs        share of player pairs ordered correctly (ties in actual points left out)
  top-N        overlap of the projected top N with the actual top N (N = 12 for QB/TE, 24 for RB/WR)
  captured     points scored by the projected top N as a share of the best possible N
and, against Sleeper, the pairs the two sources order differently: how often is each one right?
Intervals: bootstrap over position-weeks.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TOPN = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}


def week_stats(g: pd.DataFrame, col: str, ref: str | None, n: int) -> dict:
    y, p = g.pts.to_numpy(float), g[col].to_numpy(float)
    i, j = np.triu_indices(len(g), 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    ok = dy != 0
    right = (np.sign(dp[ok]) == np.sign(dy[ok]))
    top_a, top_p = set(np.argsort(-y)[:n]), set(np.argsort(-p)[:n])
    out = dict(sp=spearmanr(p, y).statistic, pair_right=right.sum(), pair_n=ok.sum(), top=len(top_a & top_p) / n, cap=y[list(top_p)].sum() / y[list(top_a)].sum(), n=len(g))
    if ref:
        dr = g[ref].to_numpy(float)[i] - g[ref].to_numpy(float)[j]
        dis = ok & (np.sign(dp) != np.sign(dr)) & (dp != 0) & (dr != 0)
        out["dis_n"], out["dis_right"] = dis.sum(), (np.sign(dp[dis]) == np.sign(dy[dis])).sum()
    return out


def evaluate(pool: pd.DataFrame, cols: dict, ref: str = "sl", boots: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in pool.groupby(["season", "week", "position"]) if len(g) >= 8]
    stats = {name: pd.DataFrame([week_stats(g, c, None if c == ref else ref, TOPN[g.position.iloc[0]]) for g in groups]) for name, c in cols.items()}
    base = stats[next(k for k, c in cols.items() if c == ref)]
    idx = rng.integers(0, len(groups), size=(boots, len(groups)))
    rows = []
    for name, s in stats.items():
        w = s.n.to_numpy(float)
        pairs = lambda t: t.pair_right.sum() / t.pair_n.sum()
        d_pairs = np.array([pairs(s.iloc[b]) - pairs(base.iloc[b]) for b in idx[:500]])
        r = dict(model=name, spearman=np.average(s.sp, weights=w), pairs=pairs(s), top_N=s.top.mean(), captured=s.cap.mean(),
                 d_pairs_vs_Sleeper=pairs(s) - pairs(base), lo=np.percentile(d_pairs, 2.5), hi=np.percentile(d_pairs, 97.5))
        if "dis_n" in s:
            b = np.array([s.dis_right.to_numpy()[k].sum() / max(s.dis_n.to_numpy()[k].sum(), 1) for k in idx])
            r.update(disagree_pairs=int(s.dis_n.sum()), right_when_disagree=s.dis_right.sum() / max(s.dis_n.sum(), 1), d_lo=np.percentile(b, 2.5), d_hi=np.percentile(b, 97.5))
        rows.append(r)
    return pd.DataFrame(rows)


def by_position(pool: pd.DataFrame, cols: dict, ref: str = "sl") -> pd.DataFrame:
    rows = []
    for pos, g in pool.groupby("position"):
        e = evaluate(g, cols, ref, boots=300)
        for _, r in e.iterrows():
            rows.append(dict(position=pos, model=r.model, spearman=r.spearman, pairs=r.pairs, top_N=r.top_N, captured=r.captured))
    return pd.DataFrame(rows)
