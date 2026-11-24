"""Do FTN's previous-season DVOA / DYAR numbers help the black-box model? Same protocol as ml_addgroup: train 2021-25, test 2026, against the working set.

    uv run python -m analysis.ml_ftn [base set, default avail+qb+cal]

Variants: + team (own offense and defense, opponent offense and defense, last season's DVOA incl. pass / rush and weighted), + player (his own last-season
DYAR / DVOA / volume / success / catch rate), + both. Inputs are last season's end-of-season numbers only (analysis/ftn.py), so nothing leaks from 2026.
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

from analysis import ftn, ml_data as MD
from analysis.ml_model import ID, POOL_MIN, feature_sets, fit_predict, metrics, vs
from analysis.ml_ranking import evaluate

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 230
    base_key = sys.argv[1] if len(sys.argv) > 1 else "avail+qb+cal"
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    base = feature_sets(F)[base_key]                       # computed before the FTN columns exist, so they cannot leak into the base set
    F = ftn.prior_season(F)
    team = [c for c in F.columns if c.startswith(("ftn_t_", "ftn_o_"))]
    player = [c for c in F.columns if c.startswith("ftn_p_")]
    variants = {"base": base, "+team": base + team, "+player": base + player, "+both": base + team + player}
    print(f"{base_key}: {len(base)} inputs; FTN team inputs {len(team)}, player inputs {len(player)}")
    tr, te = F[F.season.between(2021, 2025)], F[F.season == 2026].copy()
    P = te[ID].copy()
    for name, feats in variants.items():
        pr = fit_predict(tr, te, feats)
        for c in pr:
            P[f"{c}|{name}"] = pr[c]
        P[f"blend|{name}"] = 0.5 * P.sl + 0.5 * pr["ensemble"]
        print("trained", name, flush=True)
    P.to_pickle("/tmp/ml_ftn.pkl")
    P = P.dropna(subset=["sl"])
    pool = P[P.sl >= POOL_MIN]
    y = pool.pts.to_numpy(float)
    rows = [dict(model="Sleeper (public)", **metrics(y, pool.sl.to_numpy(float)))]
    for name in variants:
        for kind in ("ensemble", "blend"):
            c = f"{kind}|{name}"
            d, lo, hi = vs(pool, c)
            rows.append(dict(model=f"{'black box' if kind == 'ensemble' else '50-50 blend'} {name}", **metrics(y, pool[c].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi))
    print(f"\n2026 weeks {int(pool.week.min())}-{int(pool.week.max())}, n={len(pool)} player-games (Sleeper projection >= {POOL_MIN})")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    print("\nMAE change against base (negative = FTN helps), 95% interval from team-week bootstrap:")
    for name in ("+team", "+player", "+both"):
        out = []
        for kind in ("ensemble", "blend"):
            d, lo, hi = vs(pool, f"{kind}|{name}", f"{kind}|base")
            out.append(f"{kind} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {name:8s} " + "   ".join(out))
        for lab, g in [(p, pool[pool.position == p]) for p in MD.POS] + [("wk 1-4", pool[pool.week <= 4]), ("wk 5+", pool[pool.week >= 5])]:
            d, lo, hi = vs(g, f"ensemble|{name}", "ensemble|base")
            print(f"      {lab:7s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(g)})")
    rp = P[(P.sl >= POOL_MIN) | (P["ensemble|base"] >= POOL_MIN)]
    cols = {"Sleeper (public)": "sl", "black box base": "ensemble|base", "black box +both": "ensemble|+both", "blend base": "blend|base", "blend +both": "blend|+both"}
    print("\nranking, all positions:")
    print(evaluate(rp, cols, boots=500).to_string(index=False, float_format="{:.3f}".format))
