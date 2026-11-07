"""Final backtest of module 1 (nflfp/team_yards.py): rush = ridge average -> Vegas, pass = ridge opponent-adjusted -> wind -> Vegas.

Walk-forward on 2025 and 2026 (2024 is season 2025's "last season"). Compared with the league average and the team's own average to date,
and with the earlier exploratory ridge model (nflfp/yards.py).

    uv run python -m analysis.team_yards_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.yards_backtest import MAIN_FROM, load, metrics, walk_forward
from nflfp import data, team_yards as TY, weather


def load_env(seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    parts = []
    for y in seasons:
        sched = data.load_schedule(y)
        parts.append(TY.environment(sched, weather.game_weather(sched)))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    games, fixtures = load()
    env = load_env()
    cfg = TY.TeamYardsConfig()
    out = TY.walk_forward(games, env, [2024, 2025, 2026], cfg)
    base = walk_forward(games, fixtures, only=("ridge",))          # league / team-average baselines and the earlier ridge model
    base = base.rename(columns={"rush_ridge": "rush_old", "pass_ridge": "pass_old"})
    bt = base.merge(out[TY.KEYS + ["rush_ridge", "rush_exp", "pass_ridge_opp", "pass_wind", "pass_exp"]], on=TY.KEYS, how="inner")
    print(f"{len(bt)} team-games, 2025-26. rush: K={cfg.rush_strength:.0f}, last-season weight {cfg.rush_prior_weight}; pass: K={cfg.pass_strength:.0f}, last-season weight {cfg.pass_prior_weight}\n")
    rows = {"rush": {"league average to date": "rush_league", "team's own average to date": "rush_team_avg", "earlier exploratory model (ridge)": "rush_old",
                     "ridge average": "rush_ridge", "FINAL: ridge average + Vegas": "rush_exp"},
            "pass": {"league average to date": "pass_league", "team's own average to date": "pass_team_avg", "earlier exploratory model (ridge)": "pass_old",
                     "ridge opponent-adjusted": "pass_ridge_opp", "+ wind": "pass_wind", "FINAL: + Vegas": "pass_exp"}}
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), (f"weeks 1-{MAIN_FROM - 1}", bt[bt.week < MAIN_FROM]),
                       (f"2025 weeks {MAIN_FROM}+", bt[(bt.season == 2025) & (bt.week >= MAIN_FROM)]), (f"2026 weeks {MAIN_FROM}+", bt[(bt.season == 2026) & (bt.week >= MAIN_FROM)])):
        for side in ("rush", "pass"):
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games; actual mean {sub[f'{side}_yds'].mean():.1f}, sd {sub[f'{side}_yds'].std():.1f})")
            print(pd.DataFrame({k: metrics(sub, side, c) for k, c in rows[side].items()}).T.drop(columns="n").to_string()); print()
    print("fitted slopes at the end of each season: Vegas (yards per point favoured), wind (pass yards per 5 mph)")
    print(out[out.season.isin([2025, 2026])].groupby("season")[["slope_rush_vegas", "slope_pass_wind", "slope_pass_vegas"]].last().round(2).to_string())
