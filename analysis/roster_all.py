"""One table: every projection source and model for every player on a Sleeper user's roster, plus our bust/boom and ESPN's range.
Run after `python -m analysis.ml_week <user> <week>` (it leaves /tmp/ml_week_rows.pkl with every model's projection).

    uv run python -m analysis.roster_all <username> <week>
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from analysis import espn as ESPN
from analysis.ml_matchup import distributions
from nflfp import league, sleeper

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 260
    user, week = sys.argv[1], int(sys.argv[2])
    te = pd.read_pickle("/tmp/ml_week_rows.pkl")
    R = distributions(te.copy()).dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id").set_index("sleeper_id")
    dump = sleeper.load_players()
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/2026")[0]
    sc, _ = league.scoring_from_league(lg["scoring_settings"])
    me = next(r for r in sleeper.get(f"league/{lg['league_id']}/rosters") if r["owner_id"] == uid)
    mu = sleeper.get(f"league/{lg['league_id']}/matchups/{week}")
    st = set(next(x for x in mu if x["roster_id"] == me["roster_id"])["starters"])
    ids = list(dict.fromkeys(s for s in me["players"] + list(me.get("taxi") or []) if dump.get(s, {}).get("position") in ("QB", "RB", "WR", "TE")))
    nm = lambda s: f'{dump[s]["first_name"]} {dump[s]["last_name"]}'
    pl = ESPN.player_list()
    pl["key"] = pl.name.map(sleeper.norm) + "|" + pl.position
    mp = pl.drop_duplicates("key").set_index("key").espn_id
    eid = {s: mp.get(sleeper.norm(nm(s)) + "|" + dump[s]["position"]) for s in ids}
    proj = ESPN.to_points(ESPN.projections([int(v) for v in set(eid.values()) if pd.notna(v)], [202600 + week]), sc).drop_duplicates("espn_id").set_index("espn_id")
    g = lambda r, c: float(r[c]) if r is not None and pd.notna(r.get(c)) else np.nan
    rows = []
    for s in ids:
        p, r = dump[s], (R.loc[s] if s in R.index else None)
        e = proj.loc[eid[s]] if pd.notna(eid[s]) and eid[s] in proj.index else None
        ok = e is not None and e.espn_mean > 0
        rows.append(dict(player=nm(s), pos=p["position"], team=p["team"], S="S" if s in st else ("taxi" if s in (me.get("taxi") or []) else ""), status=p.get("injury_status") or "",
                         sleeper=g(r, "sl"), espn=e.espn_mean if ok else np.nan, modules=g(r, "modules"), modules_act=g(r, "modules_active"), ridge=g(r, "ridge"), trees=g(r, "trees"),
                         trees_poi=g(r, "trees_poisson"), nn=g(r, "neural_net"), ensemble=g(r, "ensemble"), blend=g(r, "blend"), stacked=g(r, "stacked"), bust=g(r, "bust"), boom=g(r, "boom"),
                         espn_bust=max(e.espn_mean - 1.645 * 1.23 * e.espn_sd, 0) if ok else np.nan, espn_boom=e.espn_mean + 1.645 * 1.23 * e.espn_sd if ok else np.nan))
    T = pd.DataFrame(rows)
    src = ["sleeper", "espn", "modules", "modules_act", "ridge", "trees", "trees_poi", "nn", "ensemble", "blend", "stacked"]
    T["avg"], T["lowest"], T["highest"] = T[src].mean(axis=1), T[src].min(axis=1), T[src].max(axis=1)
    T = T.sort_values("avg", ascending=False).reset_index(drop=True)
    print(f"{lg['name']}, week {week}\n")
    print(T[["player", "pos", "team", "S", "status"] + src + ["avg"]].to_string(float_format="{:.1f}".format))
    print()
    print(T[["player", "pos", "bust", "blend", "boom", "espn_bust", "espn", "espn_boom", "lowest", "highest"]].rename(columns={"blend": "projected", "espn": "espn_proj"}).to_string(float_format="{:.1f}".format))
    print("\nno projection:", "; ".join(f'{nm(s)} ({dump[s]["team"]}, {dump[s].get("injury_status") or "bye/no row"})' for s in ids if s not in R.index) or "none")
    T.to_pickle("/tmp/roster_all.pkl")
