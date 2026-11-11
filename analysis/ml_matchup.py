"""Monte Carlo of a head-to-head matchup: which legal lineup is most likely to win?

Every player gets a distribution (analysis/ml_dist.py: quantile models around the blended projection). Scores are drawn jointly: players on the same NFL team or
on opposite sides of the same game move together as they have since 2021 (QB-WR same team about +0.4, and so on; Gaussian copula), everyone else independently.
All legal lineups from your roster are scored against the same simulated opponent totals, so their win probabilities are directly comparable.

    uv run python -m analysis.ml_matchup <username> [week] [--taxi] [--sims 100000] [--version simple|scaled|medium]
Kickers and defenses are not modelled: each side gets the same independent draw (K 8 +/- 4, DEF 7.5 +/- 5.5), or the real score if already played.
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import ml_data as MD
from analysis import ml_dist as DI
from analysis.ml_hedge import nearest_psd, pair_corr, residuals
from analysis.ml_model import feature_sets
from analysis.ml_week import SEASON, week_rows
from nflfp import sleeper

warnings.filterwarnings("ignore")
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}


def distributions(rows: pd.DataFrame, version: str = DI.VERSION) -> pd.DataFrame:
    """Adds the quantile columns, bust / boom and `rel` (relative volatility; 1.0 = typical for his position and projection) to pre-game rows."""
    X, R = DI.prepare(rows)
    tr = X[X.season.between(*DI.TRAIN) & X.sl.notna()]
    Q = DI.fit_version(tr, version)(R)
    if "rel" not in Q:
        Q["rel"] = 1.0
    R = R.join(Q)
    R["bust"], R["boom"] = R[f"q{int(DI.LO * 100):02d}"], R[f"q{int(DI.HI * 100):02d}"]
    return R


def correlation(players: pd.DataFrame, H: pd.DataFrame) -> np.ndarray:
    """Correlation of score surprises between every pair: same NFL team, opposing teams of one game, or none."""
    n = len(players)
    C, cache = np.eye(n), {}
    pos, team, opp = players.position.to_numpy(), players.team.to_numpy(), players.opp.to_numpy()
    for i, j in itertools.combinations(range(n), 2):
        if team[i] == team[j]:
            key = (pos[i], pos[j], False)
        elif opp[i] == team[j]:
            key = (pos[i], pos[j], True)
        else:
            continue
        if key not in cache:
            cache[key] = pair_corr(H, key[0], key[1], opposite=key[2])[0]
        C[i, j] = C[j, i] = cache[key]
    return nearest_psd(C)


def legal_lineups(mine: pd.DataFrame):
    qbs = list(mine.index[mine.position == "QB"])
    sk = list(mine.index[mine.position.isin(["RB", "WR", "TE"])])
    n_sk = SLOTS["RB"] + SLOTS["WR"] + SLOTS["TE"] + SLOTS["FLEX"]
    for q in qbs:
        for combo in itertools.combinations(sk, n_sk):
            p = mine.position.loc[list(combo)]
            if (p == "RB").sum() >= SLOTS["RB"] and (p == "WR").sum() >= SLOTS["WR"] and (p == "TE").sum() >= SLOTS["TE"]:
                yield (q,) + combo


if __name__ == "__main__":
    pd.options.display.width = 230
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    user = args[0]
    week = int(args[1]) if len(args) > 1 else sleeper.get("state/nfl")["week"]
    taxi_ok = "--taxi" in sys.argv
    N = int(sys.argv[sys.argv.index("--sims") + 1]) if "--sims" in sys.argv else 100000
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else DI.VERSION
    dump = sleeper.load_players()
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/{SEASON}")[0]
    rosters = sleeper.get(f"league/{lg['league_id']}/rosters")
    me = next(r for r in rosters if r["owner_id"] == uid)
    mu = sleeper.get(f"league/{lg['league_id']}/matchups/{week}")
    mine_m = next(x for x in mu if x["roster_id"] == me["roster_id"])
    opp_m = next(x for x in mu if x["matchup_id"] == mine_m["matchup_id"] and x["roster_id"] != me["roster_id"])
    rows = distributions(week_rows(week), version).dropna(subset=["sleeper_id"]).set_index("sleeper_id")
    nm = lambda s: f'{dump.get(s, {}).get("first_name", "")} {dump.get(s, {}).get("last_name", "")}'.strip()
    skip = set(me.get("reserve") or []) | (set() if taxi_ok else set(me.get("taxi") or []))
    my_ids = [s for s in me["players"] if s in rows.index and s not in skip]
    no_row = [f"{nm(s)} ({dump.get(s, {}).get('position')} {dump.get(s, {}).get('team')}{', taxi' if s in (me.get('taxi') or []) else ''}{', reserve' if s in (me.get('reserve') or []) else ''})"
              for s in me["players"] if dump.get(s, {}).get("position") in MD.POS and s not in my_ids]
    opp_ids = [s for s in opp_m["starters"] if s in rows.index]
    opp_missing = [nm(s) or s for s in opp_m["starters"] if s not in rows.index and dump.get(s, {}).get("position") in MD.POS]
    ids = my_ids + [s for s in opp_ids if s not in my_ids]
    P = rows.loc[ids].copy()
    P["name"] = [nm(s) for s in ids]
    P["side"] = ["me"] * len(my_ids) + ["opp"] * (len(ids) - len(my_ids))
    qc = DI.qcols()
    show = P[["name", "position", "team", "opp", "side", "sl", "ensemble", "proj", "bust", "boom", "q25", "q50", "q75", "rel"]].copy()
    show["start"] = ["S" if s in set(mine_m["starters"]) | set(opp_m["starters"]) else "" for s in ids]
    print(f"{lg['name']}, week {week}. bust / boom = {int(DI.LO * 100)}th / {int(DI.HI * 100)}th percentile (a {round((DI.HI - DI.LO) * 100)}% interval); projected = blend of our ensemble and Sleeper\n")
    for side, lab in (("me", "YOUR ROSTER"), ("opp", "OPPONENT'S STARTERS")):
        d = show[show.side == side].sort_values("proj", ascending=False)
        print(lab)
        print(d[["name", "position", "team", "opp", "start", "bust", "proj", "boom", "q25", "q50", "q75", "rel", "sl", "ensemble"]].rename(
            columns={"position": "pos", "start": "S", "proj": "projected", "q25": "25th", "q50": "median", "q75": "75th", "rel": "volatility", "sl": "sleeper",
                     "ensemble": "ours"}).to_string(index=False, formatters={"volatility": "{:.2f}".format}, float_format="{:.1f}".format))
        print()
    if no_row:
        print("not available to start (bye, out, or excluded):", "; ".join(no_row))
    if opp_missing:
        print("opponent starters with no projection (counted as 0):", "; ".join(opp_missing))

    # joint simulation
    H = residuals(pd.read_pickle(MD.OUT).query("season >= 2021"))
    C = correlation(P, H)
    rng = np.random.default_rng(0)
    U = norm.cdf(rng.multivariate_normal(np.zeros(len(P)), C, size=N))
    S = pd.DataFrame(DI.sample(P[qc].to_numpy(float), U), columns=ids)
    extra = lambda mean, sd: np.maximum(rng.normal(mean, sd, N), 0)
    pts_of = lambda m: dict(zip(m["starters"], m.get("starters_points") or [0] * len(m["starters"])))

    def k_def(m):
        tot, note = np.zeros(N), []
        for s in m["starters"]:
            pos = dump.get(s, {}).get("position") or ("DEF" if s.isalpha() else None)
            if pos in ("K", "DEF"):
                played = pts_of(m).get(s, 0)
                tot += played if played else (extra(8.0, 4.0) if pos == "K" else extra(7.5, 5.5))
                note.append(f"{nm(s) or s} ({pos}{', ' + str(played) + ' scored' if played else ''})")
        return tot, note

    my_kd, my_note = k_def(mine_m)
    op_kd, op_note = k_def(opp_m)
    opp_total = S[opp_ids].sum(axis=1).to_numpy() + op_kd
    mine = P.loc[my_ids]
    Sm = S[my_ids].to_numpy()
    col = {s: i for i, s in enumerate(my_ids)}
    res = []
    for lu in legal_lineups(mine):
        t = Sm[:, [col[s] for s in lu]].sum(axis=1) + my_kd
        res.append((lu, float((t > opp_total).mean()), float(t.mean()), float(np.percentile(t, 5)), float(np.percentile(t, 95))))
    res.sort(key=lambda r: -r[1])
    best = res[0]
    cur = tuple(s for s in mine_m["starters"] if s in col)
    cur_r = next((r for r in res if set(r[0]) == set(cur)), None)
    best_mean = max(res, key=lambda r: r[2])
    tb = Sm[:, [col[s] for s in best[0]]].sum(axis=1) + my_kd
    print(f"SIMULATION: {N:,} games, {len(res)} legal lineups. Opponent: expected {opp_total.mean():.1f}, 90% range {np.percentile(opp_total, 5):.0f}-{np.percentile(opp_total, 95):.0f} "
          f"(K/DEF: {', '.join(op_note)}). Yours K/DEF: {', '.join(my_note)}\n")
    label = lambda lu: ", ".join(mine.name.loc[list(lu)])
    print("lineups most likely to win:")
    for i, (lu, pw, mu_, lo, hi) in enumerate(res[:8], 1):
        t = Sm[:, [col[s] for s in lu]].sum(axis=1) + my_kd
        d = (t > opp_total).astype(float) - (tb > opp_total).astype(float)
        diff = "" if i == 1 else f" | vs best {d.mean() * 100:+.2f} pts (+/-{2 * d.std() / np.sqrt(N) * 100:.2f})"
        swap = "" if i == 1 else "  [" + "; ".join(f"{mine.name.loc[a]} for {mine.name.loc[b]}" for a, b in zip(sorted(set(lu) - set(best[0])), sorted(set(best[0]) - set(lu)))) + "]"
        print(f"  {i}. P(win) {pw * 100:5.1f}% | expected {mu_:6.1f} | 90% range {lo:.0f}-{hi:.0f}{diff}{swap}")
        if i == 1:
            print("       " + label(lu))
    if cur_r:
        rank = res.index(cur_r) + 1
        t = Sm[:, [col[s] for s in cur_r[0]]].sum(axis=1) + my_kd
        d = (t > opp_total).astype(float) - (tb > opp_total).astype(float)
        print(f"\nyour current lineup: rank {rank} of {len(res)}, P(win) {cur_r[1] * 100:.1f}% ({d.mean() * 100:+.2f} pts vs the best, +/-{2 * d.std() / np.sqrt(N) * 100:.2f}), expected {cur_r[2]:.1f}")
        chg = [f"{mine.name.loc[a]} in for {mine.name.loc[b]}" for a, b in zip(sorted(set(best[0]) - set(cur)), sorted(set(cur) - set(best[0])))]
        print("  changes to reach the best lineup:", "; ".join(chg) if chg else "none")
    print(f"highest expected points: {best_mean[2]:.1f}, P(win) {best_mean[1] * 100:.1f}%" + ("  (same lineup as the most likely to win)" if set(best_mean[0]) == set(best[0]) else "  -> " + label(best_mean[0])))
    # each player's value: best lineup with him vs best lineup without him
    print("\neach player's effect: best P(win) among lineups that include him minus best among lineups without him")
    out = []
    for s in my_ids:
        w = max((r[1] for r in res if s in r[0]), default=np.nan)
        wo = max((r[1] for r in res if s not in r[0]), default=np.nan)
        out.append((mine.name.loc[s], mine.position.loc[s], w, wo))
    for name, pos, w, wo in sorted(out, key=lambda t: -(t[2] - t[3]) if np.isfinite(t[3]) else -9):
        print(f"  {name:20s} {pos}  with {w * 100:5.1f}% | " + (f"without {wo * 100:5.1f}% | {(w - wo) * 100:+5.1f}" if np.isfinite(wo) else "no alternative at his position"))
