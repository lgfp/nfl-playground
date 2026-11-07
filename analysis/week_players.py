"""Module 2 for one week: module 1's team yards split among players by Sleeper's projected volume.

    uv run python -m analysis.week_players [season] [week] [top_n]
"""
from __future__ import annotations

import sys

import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import attribution as A, team_yards as TY

if __name__ == "__main__":
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 220
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    games, _ = load()
    env = load_env()
    teams = TY.project_week(games, env, season, week)
    from nflfp import data
    pw = data.load_player_weeks(season)
    rates = A.league_rates(pw[pw.week < week])
    out = A.attribute(teams, A.shares(A.sleeper_volume(season, week)), rates)
    print(f"season {season} week {week}: rates {rates}\n")
    print(out.head(top)[["name", "position", "team", "opp", "carries_share", "targets_share", "rush_yds", "rec_yds", "pass_yds", "receptions", "rush_pts", "rec_pts", "pass_pts", "exp_pts"]].to_string(index=False))
    print("\nTop RB:\n", out[out.position == "RB"].head(10)[["name", "team", "opp", "carries_share", "targets_share", "rush_pts", "rec_pts", "exp_pts"]].to_string(index=False))
    print("\nteams missing volume:", sorted(set(teams.off_team) - set(out.team)))
