"""Win-probability start/sit: simulate one head-to-head matchup, choosing the lineup that maximises P(win) rather than expected points.

Each player's outcome is drawn from the real spread of results of players with a similar projection (weeks 4-8, analysis/backtest_players.py), by position group, so a
low-projection player has a right-skewed distribution (rarely booms, often a dud) and a high-projection player a tighter one. Kicker / defence: assumed normals.

    uv run python -m analysis.winprob <username> [week]
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from analysis.week_players import project
from nflfp import league, sleeper

GROUP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "WR"}
N = 40000


def sampler(pool: pd.DataFrame, pos: str, mean: float, rng, k: int = 70) -> np.ndarray:
    d = pool[pool.grp == GROUP[pos]]
    near = d.iloc[(d.blend - mean).abs().argsort()[:k]]
    shift = mean - near.blend.mean()                  # re-centre so the neighbours' average projection equals this player's
    return np.maximum(rng.choice(near.act.to_numpy(), N) + shift, 0.0)


def run(user: str, week: int, overrides: dict | None = None, seed: int = 0):
    overrides = overrides or {}
    rng = np.random.default_rng(seed)
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/{2026}")[0]
    scoring, _ = league.scoring_from_league(lg["scoring_settings"])
    full, _ = project(2026, week, scoring, active_only=True)
    f = full.set_index("sleeper_id")
    rosters = sleeper.get(f"league/{lg['league_id']}/rosters")
    me = next(r for r in rosters if r["owner_id"] == uid)
    m = next(x for x in sleeper.get(f"league/{lg['league_id']}/matchups/{week}") if x["roster_id"] == me["roster_id"])
    opp = next(x for x in sleeper.get(f"league/{lg['league_id']}/matchups/{week}") if x["matchup_id"] == m["matchup_id"] and x["roster_id"] != me["roster_id"])
    dump = sleeper.load_players()
    P = pd.read_pickle("/tmp/backtest_players.pkl")
    pool = P[P.pid.notna() & ((P.sl >= 3) | (P.ours >= 3))].copy()
    pool["blend"], pool["grp"] = (pool.ours + pool.sl) / 2, pool.position.map(GROUP)

    def draws(sid):
        pl = dump[sid]; pos = pl["position"]; name = f'{pl["first_name"]} {pl["last_name"]}'
        if pos == "K":
            return name, pos, 8.0, np.maximum(rng.normal(8.0, 3.8, N), 0)
        if pos == "DEF" or sid.isalpha() and len(sid) <= 3:
            return name, "DEF", 8.0, np.maximum(rng.normal(8.0, 4.8, N), -2)
        if sid not in f.index:
            return name, pos, 0.0, np.zeros(N)
        mean = overrides.get(name, (f.loc[sid, "total_pts"] + f.loc[sid, "sl_pts"]) / 2)
        return name, pos, mean, sampler(pool, pos, mean, rng)

    # my skill players (starters + bench; taxi cannot play), opponent's starters (played players are fixed)
    mine = {}
    for sid in me["players"]:
        if dump.get(sid, {}).get("position") in ("QB", "RB", "WR", "TE") and sid not in (me.get("taxi") or []):
            n, pos, mean, x = draws(sid); mine[n] = (pos, mean, x)
    opp_total = np.full(N, float(opp["points"]))
    opp_rows = []
    for sid, pts in zip(opp["starters"], opp.get("starters_points") or [0] * len(opp["starters"])):
        n, pos, mean, x = draws(sid)
        played = pts and pts > 0
        opp_total += (pts if played else x)
        opp_rows.append((n, pos, pts if played else mean, bool(played)))
    return me, m, mine, opp_total, opp_rows, opp, scoring


def lineups(skill: dict):
    """All sets of 7 skill players that fill RB, RB, WR, WR, TE, FLEX, FLEX (at least 2 RB, 2 WR and 1 TE)."""
    pos = {n: v[0] for n, v in skill.items()}
    for combo in itertools.combinations(pos, 7):
        c = [pos[n] for n in combo]
        if c.count("RB") >= 2 and c.count("WR") >= 2 and c.count("TE") >= 1:
            yield combo


def evaluate(user: str, week: int, label: str, overrides: dict):
    me, m, mine, opp_total, opp_rows, opp, scoring = run(user, week, overrides)
    fixed = float(m["points"])                                          # my locked points (JAX DEF)
    qb = mine["Dak Prescott"][2]
    k = np.maximum(np.random.default_rng(1).normal(8.0, 3.8, N), 0)     # Reichard (assumed)
    skill = {n: v for n, v in mine.items() if v[0] != "QB"}
    res = []
    for combo in lineups(skill):
        tot = fixed + qb + k + sum(skill[n][2] for n in combo)
        res.append((frozenset(combo), float((tot > opp_total).mean()), float(tot.mean()), float(np.percentile(tot, 90))))
    res.sort(key=lambda r: -r[1])
    print(f"\n######## {label} ########")
    print(f"opponent: {opp['points']:.1f} locked, expected final {opp_total.mean():.1f} (sd {opp_total.std():.1f}) | mine: {fixed:.1f} locked, best-lineup expected "
          f"{max(r[2] for r in res):.1f}")
    print("opponent lineup:", "; ".join(f"{n} ({p}) {v:.1f}{' played' if pl else ''}" for n, p, v, pl in opp_rows))
    print("player means used:", ", ".join(f"{n} {v[1]:.1f}" for n, v in sorted(skill.items(), key=lambda kv: -kv[1][1])))
    print("\ntop lineups by P(win):")
    for combo, pw, mu, p90 in res[:5]:
        print(f"  P(win) {pw:.3f} | mean {mu:.1f} | 90th pct {p90:.1f} | {', '.join(sorted(combo))}")
    best_mean = max(res, key=lambda r: r[2])
    print(f"best by mean: P(win) {best_mean[1]:.3f} | mean {best_mean[2]:.1f} | {', '.join(sorted(best_mean[0]))}")
    rec = frozenset(["Chase Brown", "James Cook III", "CeeDee Lamb", "Chris Olave", "Travis Kelce", "Justin Jefferson", "Devin Singletary"])
    cur = frozenset(["Jonah Coleman", "James Cook III", "CeeDee Lamb", "Chris Olave", "Mike Gesicki", "Blake Corum", "Justin Jefferson"])
    print("\nspecific lineups:")
    for lab, base in (("current lineup", cur), ("recommended (Brown, Singletary, Kelce)", rec),
                      ("recommended, Corum for Singletary", rec - {"Devin Singletary"} | {"Blake Corum"}),
                      ("recommended, Coleman for Singletary", rec - {"Devin Singletary"} | {"Jonah Coleman"}),
                      ("recommended, Corum for Jefferson", rec - {"Justin Jefferson"} | {"Blake Corum"}),
                      ("recommended, Corum for Olave", rec - {"Chris Olave"} | {"Blake Corum"}),
                      ("recommended, Gesicki for Kelce", rec - {"Travis Kelce"} | {"Mike Gesicki"})):
        r = next((x for x in res if x[0] == base), None)
        if r: print(f"  {lab:42s} P(win) {r[1]:.3f} | mean {r[2]:.1f} | 90th pct {r[3]:.1f}")
    # boom odds of single players
    print("\nsingle-player outcome odds (share of simulations):")
    for n in ("Chase Brown", "Blake Corum", "Devin Singletary", "Jonah Coleman", "Justin Jefferson", "Jalen Coker"):
        x = skill[n][2]; print(f"  {n:18s} mean {x.mean():5.1f} | >=15: {np.mean(x >= 15):.2f} | >=20: {np.mean(x >= 20):.2f} | <=5: {np.mean(x <= 5):.2f}")


if __name__ == "__main__":
    pd.options.display.width = 200
    user = sys.argv[1]
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    adj = {"Devin Singletary": 13.3}                                   # module 4 hands him almost all of NYG's rushing TDs (see the notes)
    evaluate(user, week, "model projections (Singletary TD-adjusted)", adj)
    evaluate(user, week, "hand-adjusted: Brown 12.5, Corum 10", {**adj, "Chase Brown": 12.5, "Blake Corum": 10.0})
