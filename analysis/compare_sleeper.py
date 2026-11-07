"""Module 2 vs Sleeper's own projections, without touchdowns on either side (0.1/yd, 0.5/reception).

    uv run python -m analysis.compare_sleeper [season] [week] [top_n]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import attribution as A, data, team_yards as TY
from nflfp.model import Scoring

if __name__ == "__main__":
    pd.options.display.float_format = "{:.1f}".format
    pd.options.display.width = 220
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    sc = Scoring()
    games, _ = load()
    teams = TY.project_week(games, load_env(), season, week)
    pw = data.load_player_weeks(season)
    vol = A.sleeper_volume(season, week)
    out = A.attribute(teams, A.shares(vol), A.league_rates(pw[pw.week < week]), sc)
    out["sl_rush_pts"] = sc.yard * out.sl_rush_yds
    out["sl_rec_pts"] = sc.yard * out.sl_rec_yds + sc.ppr * out.sl_receptions
    for title, ours, theirs, extra in (("RUNNERS (rushing points)", "rush_pts", "sl_rush_pts", ["carries_share", "sl_rush_yds", "rush_yds"]),
                                      ("RECEIVERS (receiving points)", "rec_pts", "sl_rec_pts", ["targets_share", "sl_rec_yds", "rec_yds"])):
        d = out.sort_values(ours, ascending=False).head(top).reset_index(drop=True)
        d["diff"] = d[ours] - d[theirs]
        d["sl_rank"] = d[theirs].rank(ascending=False).astype(int)
        d.index += 1
        print(f"\n=== TOP {top} {title}, week {week} ===")
        print(d[["name", "position", "team", "opp", ours, theirs, "diff", "sl_rank"] + extra].rename(columns={ours: "ours", theirs: "sleeper"}).to_string(float_format="{:.2f}".format))
        full = out[[ours, theirs]]
        print(f"all players: corr {full.corr().iloc[0,1]:.3f}, mean ours {full[ours].mean():.2f} vs sleeper {full[theirs].mean():.2f}; top-{top} mean diff {d['diff'].mean():+.2f}, MAD {d['diff'].abs().mean():.2f}")
        for side, tot in (("ours", ours), ("sleeper", theirs)):
            pass
    t = out.groupby("team")[["rush_yds", "sl_rush_yds", "rec_yds", "sl_rec_yds"]].sum()
    print(f"\nleague total per team: rush ours {t.rush_yds.mean():.0f} vs sleeper {t.sl_rush_yds.mean():.0f}; receiving ours {t.rec_yds.mean():.0f} vs sleeper {t.sl_rec_yds.mean():.0f}")
