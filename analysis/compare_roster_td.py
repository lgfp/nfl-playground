"""Full-points comparison (yards + receptions + TDs) for every skill player on a Sleeper user's roster: Sleeper vs ours (modules 1-4) with the shipped TD shares
and with TD shares restricted to players Sleeper projects with volume (`active_only`).

    uv run python -m analysis.compare_roster_td <username> [season] [week]
"""
from __future__ import annotations

import sys

import pandas as pd

from analysis.week_players import project
from nflfp import league, sleeper

if __name__ == "__main__":
    pd.options.display.width = 220
    user = sys.argv[1]
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    week = int(sys.argv[3]) if len(sys.argv) > 3 else sleeper.get("state/nfl")["week"]
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/{season}")[0]
    scoring, _ = league.scoring_from_league(lg["scoring_settings"])
    base, _ = project(season, week, scoring)
    act, teams = project(season, week, scoring, active_only=True)
    d = base.set_index("sleeper_id")
    a = act.set_index("sleeper_id")
    mine = next(r for r in sleeper.get(f"league/{lg['league_id']}/rosters") if r["owner_id"] == uid)
    dump = sleeper.load_players()
    rows, missing = [], []
    for sid in mine["players"]:
        pl = dump.get(sid, {})
        if pl.get("position") not in ("QB", "RB", "WR", "TE"):
            continue
        name = f'{pl.get("first_name")} {pl.get("last_name")}'
        if sid not in d.index:
            missing.append(f'{name} ({pl.get("position")} {pl.get("team")}, {pl.get("injury_status") or "no status"})')
            continue
        r, q = d.loc[sid], a.loc[sid]
        rows.append(dict(player=name, pos=pl["position"], team=r.team, opp=r.opp, status=pl.get("injury_status") or "",
                         ours=r.total_pts, ours_active=q.total_pts, sleeper=r.sl_pts, diff=q.total_pts - r.sl_pts,
                         yd_pts=r.exp_pts, td_ours=r.td_pts, td_active=q.td_pts, td_sleeper=r.sl_pts - (scoring.yard * (r.sl_rush_yds + r.sl_rec_yds) + scoring.ppr * r.sl_receptions + scoring.pass_yard * r.sl_pass_yds),
                         rush_td=q.rush_tds, rec_td=q.rec_tds, pass_td=q.pass_tds, sl_rush_td=r.sl_rush_tds, sl_rec_td=r.sl_rec_tds))
    t = pd.DataFrame(rows).sort_values("ours_active", ascending=False).reset_index(drop=True)
    print(f"week {week}, {lg['name']}, scoring {scoring}\n")
    print(t.to_string(float_format="{:.2f}".format))
    print(f"\nmean: ours {t.ours.mean():.2f} | ours active-only {t.ours_active.mean():.2f} | Sleeper {t.sleeper.mean():.2f}; TD points: ours {t.td_ours.mean():.2f} | active-only {t.td_active.mean():.2f} | Sleeper {t.td_sleeper.mean():.2f}")
    print(f"corr(active-only, Sleeper) {t.ours_active.corr(t.sleeper):.3f}; mean abs diff {t['diff'].abs().mean():.2f}")
    print("no projection:", "; ".join(missing) or "none")
