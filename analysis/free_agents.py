"""Free agents (on nobody's roster in a Sleeper league) against your own players: projection, bust and boom, for an upcoming week.

    uv run python -m analysis.free_agents <username> [week]
A free agent is a 'straight upgrade' over one of your players when his projection AND his boom (95th percentile) are both higher, and a 'boom upgrade' when only the boom is.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_matchup import distributions
from analysis.ml_week import SEASON, week_rows
from nflfp import sleeper

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 230
    user = sys.argv[1]
    week = int(sys.argv[2]) if len(sys.argv) > 2 else sleeper.get("state/nfl")["week"]
    dump = sleeper.load_players()
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/{SEASON}")[0]
    rosters = sleeper.get(f"league/{lg['league_id']}/rosters")
    me = next(r for r in rosters if r["owner_id"] == uid)
    taken = set()
    for r in rosters:
        taken |= set(r.get("players") or []) | set(r.get("taxi") or []) | set(r.get("reserve") or [])
    mu = sleeper.get(f"league/{lg['league_id']}/matchups/{week}")
    starters = set(next(x for x in mu if x["roster_id"] == me["roster_id"])["starters"])
    R = distributions(week_rows(week, refresh=True)).dropna(subset=["sleeper_id"])
    R["name"] = [f'{dump.get(s, {}).get("first_name", "")} {dump.get(s, {}).get("last_name", "")}'.strip() or n for s, n in zip(R.sleeper_id, R.player_display_name)]
    R["status"] = [dump.get(s, {}).get("injury_status") or "" for s in R.sleeper_id]
    mine = R[R.sleeper_id.isin(set(me["players"]) | set(me.get("taxi") or []))].copy()
    mine["S"] = np.where(mine.sleeper_id.isin(starters), "S", "")
    fa = R[~R.sleeper_id.isin(taken)].copy()
    print(f"{lg['name']}, week {week}: {len(rosters)} rosters, {len(taken)} rostered players; {len(fa)} projected free agents (not Out/IR)\n")
    cols = ["name", "position", "team", "opp", "status", "bust", "proj", "boom", "sl", "ensemble"]
    for pos in MD.POS:
        m = mine[mine.position == pos].sort_values("proj", ascending=False)
        f = fa[fa.position == pos].sort_values("proj", ascending=False).head(8 if pos != "QB" else 4)
        print(f"===== {pos} =====")
        print("yours:")
        print(m[["S"] + cols].rename(columns={"sl": "sleeper", "ensemble": "ours"}).to_string(index=False, float_format="{:.1f}".format))
        print("best free agents by projection:")
        print(f[cols].rename(columns={"sl": "sleeper", "ensemble": "ours"}).to_string(index=False, float_format="{:.1f}".format))
        ups = []
        for _, a in fa[fa.position == pos].iterrows():
            for _, b in m.iterrows():
                if a.proj > b.proj and a.boom >= b.boom and a.bust >= b.bust - 0.5:
                    ups.append((a["name"], b["name"], "STRAIGHT", a.proj - b.proj, a.boom - b.boom))
                elif a.boom > b.boom + 1.0 and a.proj > b.proj - 1.0:
                    ups.append((a["name"], b["name"], "boom", a.proj - b.proj, a.boom - b.boom))
        if ups:
            u = pd.DataFrame(ups, columns=["free agent", "over", "kind", "d_proj", "d_boom"]).sort_values(["kind", "d_proj"], ascending=[False, False])
            print("upgrades (STRAIGHT = higher projection, boom and bust; boom = boom 1+ higher and projection within 1):")
            print(u.head(10).to_string(index=False, float_format="{:+.1f}".format))
        else:
            print("no free agent beats any of yours on projection with an equal or higher boom")
        print()
    # also flex-eligible: any RB/WR/TE free agent vs your weakest starting flex
    flex = mine[mine.position.isin(["RB", "WR", "TE"])].sort_values("proj")
    print("your lowest-projected RB/WR/TE (flex candidates):", ", ".join(f"{r['name']} {r.proj:.1f}" for _, r in flex.head(5).iterrows()))
    top = fa[fa.position.isin(["RB", "WR", "TE"])].sort_values("proj", ascending=False).head(10)
    print("best RB/WR/TE free agents overall:", ", ".join(f"{r['name']} ({r.position}) {r.proj:.1f}/{r.boom:.1f}" for _, r in top.iterrows()))
    fa.to_pickle("/tmp/free_agents.pkl"); mine.to_pickle("/tmp/my_players.pkl")
