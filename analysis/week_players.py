"""Modules 1-4 for one week: team yards (1), team TDs (3), players from Sleeper's projected volume (2) and play-by-play history (4); Sleeper's own points for comparison.

    uv run python -m analysis.week_players [season] [week] [top_n] [username]
Scoring comes from the Sleeper league of `username` if given, else 0.5 PPR / 0.1 per yard / 6 per TD / 0.04 per pass yard / 4 per pass TD.
"""
from __future__ import annotations

import sys

import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import attribution as A, data, league, sleeper, td_attribution as TA, team_tds as M3, team_yards as TY, tds as T
from nflfp.model import Scoring


def project(season: int, week: int, scoring: Scoring = Scoring()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player table, team table) for the week."""
    games, _ = load()
    teams = TY.project_week(games, load_env(), season, week)                                       # module 1
    plays = T.with_xtd(T.load_plays([season - 1, season]))
    yards = TY.walk_forward(games, load_env(), [season - 1, season], last_week=week)
    imp = pd.concat([M3.implied_totals(data.load_schedule(y)) for y in (season - 1, season)], ignore_index=True)
    tteams = M3.project_week(T.team_game_tds(plays), yards, imp, season, week)                    # module 3
    pw = data.load_player_weeks(season)
    vol = A.sleeper_volume(season, week)
    m2 = A.attribute(teams, A.shares(vol), A.league_rates(pw[pw.week < week]), scoring)           # module 2
    known = pw.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    dump = sleeper.load_players()
    out = sleeper.unavailable(dump, known).player_id                                              # Out / IR
    td = TA.attribute(tteams, TA.shares(plays, season, week, exclude=set(out)))                   # module 4
    id_map = sleeper.map_ids(dump, known, m2.sleeper_id)
    full = TA.add_touchdowns(m2, td, id_map, scoring)
    full["sl_pts"] = (scoring.yard * (full.sl_rush_yds + full.sl_rec_yds) + scoring.ppr * full.sl_receptions + scoring.pass_yard * full.sl_pass_yds
                      + scoring.td * (full.sl_rush_tds + full.sl_rec_tds) + scoring.pass_td * full.sl_pass_tds)
    return full, tteams


if __name__ == "__main__":
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 240
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    scoring = Scoring()
    if len(sys.argv) > 4:
        lg = sleeper.get(f"user/{sleeper.user_id(sys.argv[4])}/leagues/nfl/{season}")[0]
        scoring, _ = league.scoring_from_league(lg["scoring_settings"])
    full, teams = project(season, week, scoring)
    cols = ["name", "position", "team", "opp", "rush_yds", "rec_yds", "pass_yds", "receptions", "rush_tds", "rec_tds", "pass_tds", "exp_pts", "td_pts", "total_pts", "sl_pts"]
    print(f"season {season} week {week}, scoring {scoring}\n")
    print(full.sort_values("total_pts", ascending=False).head(top)[cols].to_string(index=False))
    print("\nteam TDs:", teams[["off_team", "def_team", "implied", "rush_td_exp", "pass_td_exp"]].sort_values("implied", ascending=False).head(6).to_string(index=False))
    print(f"\nleague-wide check: our mean team TDs {teams.total_td_exp.mean():.2f} rush+pass; players with no TD history (new/traded): "
          f"{(full[['rush_tds', 'rec_tds']].sum(axis=1) == 0).sum()} of {len(full)}")
