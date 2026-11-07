"""Backtest of module 1 (nflfp/yards.py): expected team rushing and passing yards, walk-forward on 2025 and 2026.

Every week is projected from the games before it (plus last season, for the ridge target and the Vegas fit).
Season 2025 uses 2024 as its "previous season", 2026 uses 2025.

    uv run python analysis/yards_backtest.py
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from nflfp import data, yards as Y

SEASONS = [2025, 2026]
MAIN_FROM = 4          # the headline numbers use weeks 4+ (weeks 1-3 are shown separately: that is where the ridge prior matters)


def load():
    pw = pd.concat([data.load_player_weeks(y) for y in (2024, *SEASONS)], ignore_index=True)
    games = Y.team_game_yards(pw)
    fixtures = pd.concat([Y.fixtures_from_schedule(data.load_schedule(y)) for y in (2024, *SEASONS)], ignore_index=True)
    return games, fixtures


def walk_forward(games, fixtures, cfg: Y.YardsConfig = Y.YardsConfig(), seasons=SEASONS, only=None) -> pd.DataFrame:
    rows = []
    for season in seasons:
        ctx = Y.season_context(games, fixtures, season, cfg)
        prev = games[games.season == season - 1]
        prev_avg = prev.groupby("off_team")[["rush_yds", "pass_yds"]].mean()
        for week in sorted(games.loc[games.season == season, "week"].unique()):
            out = Y.project_week(games, fixtures, season, week, cfg, ctx, only)
            cur = games[(games.season == season) & (games.week < week)]
            act = games[(games.season == season) & (games.week == week)][["off_team", "def_team", "rush_yds", "pass_yds"]]
            m = out.merge(act, on=["off_team", "def_team"], how="inner")
            own = cur.groupby("off_team")[["rush_yds", "pass_yds"]].mean()
            lg = cur[["rush_yds", "pass_yds"]].mean() if len(cur) else prev[["rush_yds", "pass_yds"]].mean()
            for s in Y.SIDES:
                m[f"{s}_league"] = lg[f"{s}_yds"]
                m[f"{s}_team_avg"] = m.off_team.map(own[f"{s}_yds"]).fillna(m.off_team.map(prev_avg[f"{s}_yds"])).fillna(lg[f"{s}_yds"])
                m[f"{s}_prev_avg"] = m.off_team.map(prev_avg[f"{s}_yds"]).fillna(lg[f"{s}_yds"])
            for base in ("plain", "ridge"):
                a = out.attrs["lines"].get(base)
                m[f"lines_rush_{base}"], m[f"lines_pass_{base}"] = a.as_slopes() if a else (np.nan, np.nan)
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def metrics(bt: pd.DataFrame, side: str, col: str) -> dict:
    a, p = bt[f"{side}_yds"].to_numpy(float), bt[col].to_numpy(float)
    ok = ~np.isnan(p)
    a, p, w = a[ok], p[ok], bt[["season", "week"]].to_numpy()[ok]
    e = p - a
    conc = tot = 0
    for key in {tuple(k) for k in w}:
        idx = (w[:, 0] == key[0]) & (w[:, 1] == key[1])
        dp, da = np.sign(p[idx][:, None] - p[idx][None, :]), np.sign(a[idx][:, None] - a[idx][None, :])
        m = np.triu(np.ones((idx.sum(),) * 2), 1) > 0
        m &= (dp != 0) & (da != 0)
        conc += int((dp[m] == da[m]).sum()); tot += int(m.sum())
    return dict(n=len(a), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(),
                R2=1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum(), corr=np.corrcoef(a, p)[0, 1], pairs=conc / tot if tot else np.nan)


def table(bt: pd.DataFrame, side: str, label: str) -> pd.DataFrame:
    cols = {"league average to date": f"{side}_league", "team's own average to date": f"{side}_team_avg", "team's previous-season average": f"{side}_prev_avg",
            **{v: f"{side}_{v}" for v, _, _ in Y.VARIANTS}}
    return pd.DataFrame({name: metrics(bt, side, c) for name, c in cols.items()}).T


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    games, fixtures = load()
    cfg = Y.YardsConfig()
    bt = walk_forward(games, fixtures, cfg)
    print(f"{len(bt)} team-games, seasons {SEASONS}; defaults: ridge strength {cfg.ridge_strength}, last-season weight {cfg.prior_weight}, "
          f"prior basis {cfg.prior_basis}, Vegas mode {cfg.lines_mode}\n")
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), ("weeks 1-3", bt[bt.week < MAIN_FROM])):
        for side in Y.SIDES:
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games; actual mean {sub[f'{side}_yds'].mean():.1f}, sd {sub[f'{side}_yds'].std():.1f})")
            print(table(sub, side, label).to_string())
            print()
    for s in SEASONS:
        sub = bt[(bt.season == s) & (bt.week >= MAIN_FROM)]
        print(f"--- {s} weeks {MAIN_FROM}+: R2  " + " | ".join(f"{side}: " + ", ".join(f"{v} {metrics(sub, side, f'{side}_{v}')['R2']:.3f}" for v, _, _ in Y.VARIANTS) for side in Y.SIDES))
    b = bt[bt.week >= MAIN_FROM].groupby(["season", "week"])[["lines_rush_ridge", "lines_pass_ridge"]].first().dropna()
    print(f"\nfitted Vegas adjustment ({cfg.lines_mode}) of the ridge model, yards per point favoured (rush, pass), selected weeks:")
    print(b.loc[[k for k in b.index if k[1] in (4, 8, 12, 16)]].round(2).T.to_string())
