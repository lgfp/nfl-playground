"""Monte Carlo of a head-to-head matchup: which legal lineup is most likely to win?

Every player gets a distribution (analysis/ml_dist.py: quantile models around the blended projection). Scores are drawn jointly: players on the same NFL team or
on opposite sides of the same game move together as they have since 2021 (QB-WR same team about +0.4, and so on; Gaussian copula), everyone else independently.
All legal lineups from your roster are scored against the same simulated opponent totals, so their win probabilities are directly comparable.

    uv run python -m analysis.ml_matchup <username> [week] [--taxi] [--sims 100000] [--version simple|scaled|medium|components] [--espn]
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
    for c in ("rel", "td_exp", "td_share"):
        if c not in Q:
            Q[c] = np.nan
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
    show = P[["name", "position", "team", "opp", "side", "sl", "ensemble", "proj", "bust", "boom", "q25", "q50", "q75", "rel", "td_exp", "td_share"]].copy()
    show["start"] = ["S" if s in set(mine_m["starters"]) | set(opp_m["starters"]) else "" for s in ids]
    print(f"{lg['name']}, week {week}. bust / boom = {int(DI.LO * 100)}th / {int(DI.HI * 100)}th percentile (a {round((DI.HI - DI.LO) * 100)}% interval); projected = blend of our ensemble and Sleeper\n")
    for side, lab in (("me", "YOUR ROSTER"), ("opp", "OPPONENT'S STARTERS")):
        d = show[show.side == side].sort_values("proj", ascending=False)
        print(lab)
        extra = ["td_exp", "td_share"] if d.td_exp.notna().any() else ["rel"]
        print(d[["name", "position", "team", "opp", "start", "bust", "proj", "boom", "q25", "q50", "q75"] + extra + ["sl", "ensemble"]].rename(
            columns={"position": "pos", "start": "S", "proj": "projected", "q25": "25th", "q50": "median", "q75": "75th", "rel": "volatility", "td_exp": "exp_TDs",
                     "sl": "sleeper", "ensemble": "ours"}).to_string(index=False, formatters={"volatility": "{:.2f}".format, "exp_TDs": "{:.2f}".format, "td_share": "{:.2f}".format},
                                                                    float_format="{:.1f}".format))
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
    cur = tuple(s for s in mine_m["starters"] if s in set(my_ids))
    lineups = list(legal_lineups(P.loc[my_ids]))
    label = lambda lu: ", ".join(P.name.loc[list(lu)])

    def run(Qm: np.ndarray, title: str, verbose: bool):
        S = pd.DataFrame(DI.sample(Qm, U), columns=ids)
        opp_total = S[opp_ids].sum(axis=1).to_numpy() + op_kd
        mine = P.loc[my_ids]
        Sm = S[my_ids].to_numpy()
        col = {s: i for i, s in enumerate(my_ids)}
        res = []
        for lu in lineups:
            t = Sm[:, [col[s] for s in lu]].sum(axis=1) + my_kd
            res.append((lu, float((t > opp_total).mean()), float(t.mean()), float(np.percentile(t, 5)), float(np.percentile(t, 95))))
        res.sort(key=lambda r: -r[1])
        best = res[0]
        tb = Sm[:, [col[s] for s in best[0]]].sum(axis=1) + my_kd
        cur_r = next((r for r in res if set(r[0]) == set(cur)), None)
        print(f"\n=== {title}: {N:,} games, {len(res)} legal lineups. Opponent expected {opp_total.mean():.1f} (90% range {np.percentile(opp_total, 5):.0f}-{np.percentile(opp_total, 95):.0f}) "
              f"| K/DEF: theirs {', '.join(op_note)}; yours {', '.join(my_note)}")
        print("lineups most likely to win:")
        for i, (lu, pw, mu_, lo, hi) in enumerate(res[:(8 if verbose else 4)], 1):
            t = Sm[:, [col[s] for s in lu]].sum(axis=1) + my_kd
            d = (t > opp_total).astype(float) - (tb > opp_total).astype(float)
            diff = "" if i == 1 else f" | vs best {d.mean() * 100:+.2f} pts (+/-{2 * d.std() / np.sqrt(N) * 100:.2f})"
            swap = "" if i == 1 else "  [" + "; ".join(f"{mine.name.loc[a]} for {mine.name.loc[b]}" for a, b in zip(sorted(set(lu) - set(best[0])), sorted(set(best[0]) - set(lu)))) + "]"
            print(f"  {i}. P(win) {pw * 100:5.1f}% | expected {mu_:6.1f} | 90% range {lo:.0f}-{hi:.0f}{diff}{swap}")
            if i == 1:
                print("       " + label(lu))
        if cur_r:
            t = Sm[:, [col[s] for s in cur_r[0]]].sum(axis=1) + my_kd
            d = (t > opp_total).astype(float) - (tb > opp_total).astype(float)
            chg = [f"{mine.name.loc[a]} in for {mine.name.loc[b]}" for a, b in zip(sorted(set(best[0]) - set(cur)), sorted(set(cur) - set(best[0])))]
            print(f"your current lineup: rank {res.index(cur_r) + 1} of {len(res)}, P(win) {cur_r[1] * 100:.1f}% ({d.mean() * 100:+.2f} vs the best, +/-{2 * d.std() / np.sqrt(N) * 100:.2f}); "
                  f"changes to reach the best: {'; '.join(chg) if chg else 'none'}")
        if verbose:
            print("each player's effect: best P(win) among lineups that include him minus best among lineups without him")
            for s in my_ids:
                w = max((r[1] for r in res if s in r[0]), default=np.nan)
                wo = max((r[1] for r in res if s not in r[0]), default=np.nan)
                print(f"  {mine.name.loc[s]:20s} {mine.position.loc[s]}  with {w * 100:5.1f}% | " + (f"without {wo * 100:5.1f}% | {(w - wo) * 100:+5.1f}" if np.isfinite(wo) else "no alternative at his position"))
        return res

    Q_ours = P[qc].to_numpy(float)
    run(Q_ours, "OURS (distribution from our model)", True)
    if "--espn" in sys.argv:
        from analysis import espn as ESPN
        from nflfp import league
        scoring, _ = league.scoring_from_league(lg["scoring_settings"])
        K_ESPN = 1.23                               # ESPN's stated sd x 1.23 gives 90% coverage on 2025 weeks 10-18 (analysis/espn_calibration.py)
        pl = ESPN.player_list()
        pl["key"] = pl.name.map(sleeper.norm) + "|" + pl.position
        P["key"] = P.name.map(sleeper.norm) + "|" + P.position
        uniq = pl.groupby("key").espn_id.nunique()
        mp = pl[pl.key.isin(uniq[uniq == 1].index)].drop_duplicates("key").set_index("key").espn_id
        P["espn_id"] = P.key.map(mp)
        proj = ESPN.to_points(ESPN.projections([int(i) for i in P.espn_id.dropna().unique()], [SEASON * 100 + week]), scoring).drop_duplicates("espn_id").set_index("espn_id")
        P["espn_mean"], P["espn_sd"] = P.espn_id.map(proj.espn_mean), P.espn_id.map(proj.espn_sd)
        ok = P.espn_mean.notna() & (P.espn_mean > 0)
        print(f"\nESPN: projections found for {int(ok.sum())} of {len(P)} players ({', '.join(P.name[~ok])}: not projected, ours used)")
        # shape from our distribution, standardized, then moved to ESPN's mean and (calibrated) width
        mean_o = P.mean_.to_numpy(float) if P.mean_.notna().all() else DI.dist_mean(Q_ours)
        sd_o = P.sd.to_numpy(float) if "sd" in P and P.sd.notna().all() else np.maximum((P.boom.to_numpy(float) - P.bust.to_numpy(float)) / 3.29, 0.5)
        Z = (Q_ours - mean_o[:, None]) / sd_o[:, None]

        def build(mean, sd):
            Q = np.sort(np.maximum(mean[:, None] + sd[:, None] * Z, 0.0), axis=1)
            return Q
        e_mean = np.where(ok, P.espn_mean, mean_o)
        e_sd = np.where(ok, P.espn_sd * K_ESPN, sd_o)
        Q_espn = build(e_mean, e_sd)
        three = np.where(ok, (P.espn_mean.fillna(0) + P.sl.fillna(P.ensemble) + P.ensemble) / 3, mean_o)
        Q_cons = build(three, 0.5 * sd_o + 0.5 * e_sd)
        cmp = P[["name", "position", "proj", "bust", "boom"]].copy()
        cmp["espn_mean"], cmp["espn_bust"], cmp["espn_boom"] = e_mean, np.maximum(e_mean - 1.645 * e_sd, 0), e_mean + 1.645 * e_sd
        print("\nours vs ESPN (ESPN range = its mean +/- 1.645 x calibrated sd; ours = 5th-95th percentile):")
        print(cmp[cmp.index.isin(my_ids)].sort_values("proj", ascending=False).rename(columns={"proj": "ours", "bust": "our_bust", "boom": "our_boom"}).to_string(index=False, float_format="{:.1f}".format))
        run(Q_espn, "ESPN's mean and range (our distribution shape)", False)
        run(Q_cons, "CONSENSUS: mean of ESPN, Sleeper and our ensemble; width halfway between ours and ESPN's", False)
