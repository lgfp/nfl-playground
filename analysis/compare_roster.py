"""Module 2 vs Sleeper's own projections (no TDs on either side) for every skill player on a Sleeper user's roster.

    uv run python -m analysis.compare_roster <username> [season] [week]
"""
from __future__ import annotations

import sys

import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import attribution as A, data, league, sleeper, team_yards as TY

if __name__ == "__main__":
    pd.options.display.width = 220
    user = sys.argv[1]
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    week = int(sys.argv[3]) if len(sys.argv) > 3 else sleeper.get("state/nfl")["week"]
    lg = sleeper.get(f"user/{sleeper.user_id(user)}/leagues/nfl/{season}")[0]
    scoring, _ = league.scoring_from_league(lg["scoring_settings"])
    games, _ = load()
    teams = TY.project_week(games, load_env(), season, week)
    pw = data.load_player_weeks(season)
    vol = A.sleeper_volume(season, week)
    out = A.attribute(teams, A.shares(vol), A.league_rates(pw[pw.week < week]), scoring).set_index("sleeper_id")
    out["sl_rush_pts"] = scoring.yard * out.sl_rush_yds
    out["sl_rec_pts"] = scoring.yard * out.sl_rec_yds + scoring.ppr * out.sl_receptions
    out["sl_pass_pts"] = scoring.pass_yard * out.sl_pass_yds
    out["sl_pts"] = out.sl_rush_pts + out.sl_rec_pts + out.sl_pass_pts
    status = vol.set_index("sleeper_id").injury_status
    mine = next(r for r in sleeper.get(f"league/{lg['league_id']}/rosters") if r["owner_id"] == sleeper.user_id(user))
    players = sleeper.load_players()
    rows, missing = [], []
    for sid in mine["players"]:
        pl = players.get(sid, {})
        if pl.get("position") not in ("QB", "RB", "WR", "TE"):
            continue
        name = f'{pl.get("first_name")} {pl.get("last_name")}'
        if sid in out.index:
            r = out.loc[sid]
            rows.append(dict(player=name, pos=pl["position"], team=r.team, opp=r.opp, status=status.get(sid) or "",
                             c_share=r.carries_share, t_share=r.targets_share, ours=r.exp_pts, sleeper=r.sl_pts, diff=r.exp_pts - r.sl_pts,
                             rush=r.rush_pts, sl_rush=r.sl_rush_pts, rec=r.rec_pts, sl_rec=r.sl_rec_pts, pas=r.pass_pts, sl_pas=r.sl_pass_pts))
        else:
            missing.append(f'{name} ({pl.get("position")} {pl.get("team")}, {pl.get("injury_status") or "no status"})')
    d = pd.DataFrame(rows).sort_values("ours", ascending=False).reset_index(drop=True)
    print(f"week {week}, {lg['name']}: scoring {scoring}\n")
    print(d.to_string(float_format="{:.2f}".format))
    print(f"\ncorr {d.ours.corr(d.sleeper):.3f}; mean diff (ours - sleeper) {d['diff'].mean():+.2f}; mean abs diff {d['diff'].abs().mean():.2f}")
    print("no projection (out/IR/bye/taxi/no volume):", "; ".join(missing) or "none")
