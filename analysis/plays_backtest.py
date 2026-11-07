"""Module 1b (play-level, nflfp/plays.py) against module 1 (game-level, nflfp/yards.py): walk-forward on 2025 and 2026.

Both are scored on the same actual team yards (from the player stats). Season 2025 uses 2024 as its previous season, 2026 uses 2025.

    uv run python -m analysis.plays_backtest
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from analysis.yards_backtest import MAIN_FROM, SEASONS, load, metrics, walk_forward
from nflfp import data, plays as P, yards as Y

KEYS = ["season", "week", "off_team", "def_team"]


def load_plays() -> pd.DataFrame:
    return P.team_game_plays(pd.concat([data.load_pbp(y) for y in (2024, *SEASONS)], ignore_index=True))


def walk_forward_plays(gp: pd.DataFrame, fixtures: pd.DataFrame, cfg: P.PlaysConfig = P.PlaysConfig(), seasons=SEASONS, only=None) -> pd.DataFrame:
    rows = []
    for season in seasons:
        ctx = P.season_context(gp, fixtures, season, cfg)
        for week in sorted(gp.loc[gp.season == season, "week"].unique()):
            out = P.project_week(gp, fixtures, season, week, cfg, ctx, only)
            for base in ("plain", "ridge"):
                a = out.attrs["lines"].get(base)
                out[f"lines_rush_{base}"], out[f"lines_pass_{base}"] = a.as_slopes() if a else (np.nan, np.nan)
            rows.append(out)
    d = pd.concat(rows, ignore_index=True).drop(columns="fav")
    return d.rename(columns={c: f"p_{c}" for c in d.columns if c not in KEYS})


def combined(gy, gp, fixtures, ycfg=Y.YardsConfig(), pcfg=P.PlaysConfig(), only=None) -> pd.DataFrame:
    bt = walk_forward(gy, fixtures, ycfg, only=only)
    p = walk_forward_plays(gp, fixtures, pcfg, only=only)
    act = gp[KEYS + ["rush_att", "pass_att", "rush_yds", "pass_yds"]].rename(columns={"rush_yds": "rush_yds_pbp", "pass_yds": "pass_yds_pbp", "rush_att": "rush_att_act", "pass_att": "pass_att_act"})
    return bt.merge(p, on=KEYS, how="left").merge(act, on=KEYS, how="left")


def block(bt: pd.DataFrame, side: str) -> pd.DataFrame:
    cols = {"league average to date": f"{side}_league", "team's own average to date": f"{side}_team_avg",
            **{f"1  game-level  {v}": f"{side}_{v}" for v, _, _ in Y.VARIANTS}, **{f"1b play-level  {v}": f"p_{side}_{v}" for v, _, _ in Y.VARIANTS}}
    return pd.DataFrame({k: metrics(bt, side, c) for k, c in cols.items()}).T


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 230
    gy, fixtures = load()
    gp = load_plays()
    m = gy.merge(gp, on=KEYS, suffixes=("", "_pbp"))
    print(f"pbp-derived vs player-stat team yards, {len(m)} team-games: rush MAE {np.abs(m.rush_yds - m.rush_yds_pbp).mean():.1f} yds, pass MAE {np.abs(m.pass_yds - m.pass_yds_pbp).mean():.1f} yds "
          f"(means {m.rush_yds.mean():.1f} / {m.rush_yds_pbp.mean():.1f} and {m.pass_yds.mean():.1f} / {m.pass_yds_pbp.mean():.1f})")
    bt = combined(gy, gp, fixtures)
    print(f"\n{len(bt)} team-games; module 1b defaults: strength volume {P.PlaysConfig().strength_volume}, efficiency {P.PlaysConfig().strength_efficiency}, "
          f"last-season weight {P.PlaysConfig().prior_weight}, Vegas on volume: {P.PlaysConfig().lines_mode}\n")
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), (f"weeks 1-{MAIN_FROM - 1}", bt[bt.week < MAIN_FROM])):
        for side in Y.SIDES:
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games)")
            print(block(sub, side).to_string()); print()
    print("--- R2 by season, weeks 4+ (module 1 ridge | 1b ridge | 1 ridge+Vegas | 1b ridge+Vegas)")
    for s in SEASONS:
        sub = bt[(bt.season == s) & (bt.week >= MAIN_FROM)]
        print(f"{s}: " + " ; ".join(f"{side}: " + " | ".join(f"{metrics(sub, side, c.format(side=side))['R2']:.3f}" for c in ("{side}_ridge", "p_{side}_ridge", "{side}_ridge_vegas", "p_{side}_ridge_vegas")) for side in Y.SIDES))

    sub = bt[bt.week >= MAIN_FROM].copy()
    print("\n=== What module 1b gets right: volume vs efficiency (ridge, no Vegas, weeks 4+)")
    diag = {}
    for side in Y.SIDES:
        sub[f"{side}_ypp_act"] = sub[f"{side}_yds_pbp"] / sub[f"{side}_att_act"]
        for q, act in ((f"{side}_att", f"{side}_att_act"), (f"{side}_ypc" if side == "rush" else "pass_ypa", f"{side}_ypp_act")):
            a, p = sub[act].to_numpy(float), sub[f"p_{q}_ridge"].to_numpy(float)
            ok = ~np.isnan(a) & ~np.isnan(p)
            diag[q] = dict(actual_mean=a[ok].mean(), actual_sd=a[ok].std(), MAE=np.abs(p[ok] - a[ok]).mean(), R2=1 - ((p[ok] - a[ok]) ** 2).sum() / ((a[ok] - a[ok].mean()) ** 2).sum(), corr=np.corrcoef(a[ok], p[ok])[0, 1])
    print(pd.DataFrame(diag).T.to_string())

    print("\n=== Vegas on volume (module 1b ridge): modes, weeks 4+")
    rows = []
    for mode in ("transfer", "relative"):
        b = combined(gy, gp, fixtures, pcfg=P.PlaysConfig(lines_mode=mode), only=("ridge", "ridge_vegas")); s2 = b[b.week >= MAIN_FROM]
        r = {"mode": mode}
        for side in Y.SIDES:
            base, v = metrics(s2, side, f"p_{side}_ridge"), metrics(s2, side, f"p_{side}_ridge_vegas")
            r[f"{side} R2 no-Vegas"], r[f"{side} R2 Vegas"], r[f"{side} MAE no-Vegas"], r[f"{side} MAE Vegas"] = base["R2"], v["R2"], base["MAE"], v["MAE"]
        last = s2.groupby(["season", "week"])[["p_lines_rush_ridge", "p_lines_pass_ridge"]].first().dropna().tail(1).iloc[0]
        r["latest slopes (rush, pass)"] = f"{last.iloc[0]:+.3f}, {last.iloc[1]:+.3f}"
        rows.append(r)
    print(pd.DataFrame(rows).set_index("mode").T.to_string())

    print("\n=== Ridge strengths for module 1b (no Vegas): sum of rush+pass R2, weeks 4+  [rows: volume strength, columns: efficiency strength]")
    res = {}
    for kv, ke in itertools.product((2, 4, 8, 16, 32), (4, 8, 16, 32, 64)):
        b = combined(gy, gp, fixtures, pcfg=P.PlaysConfig(strength_volume=kv, strength_efficiency=ke), only=("ridge",)); s3 = b[b.week >= MAIN_FROM]
        res[(kv, ke)] = (metrics(s3, "rush", "p_rush_ridge")["R2"], metrics(s3, "pass", "p_pass_ridge")["R2"])
    g = pd.DataFrame({k: {"rush": v[0], "pass": v[1], "sum": v[0] + v[1]} for k, v in res.items()}).T
    print(g["sum"].unstack().to_string()); best = g["sum"].idxmax(); print(f"best: volume {best[0]}, efficiency {best[1]} -> rush R2 {g.loc[best,'rush']:.3f}, pass R2 {g.loc[best,'pass']:.3f}")
