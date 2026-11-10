"""Two questions on the 5-season black-box model (trained 2021-25, tested on 2026): (1) do quarterback inputs help? (2) how well does it RANK players vs Sleeper?

    uv run python -m analysis.ml_qb_ranking
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import ID, POOL_MIN, feature_sets, fit_predict, metrics, vs
from analysis.ml_ranking import by_position, evaluate

warnings.filterwarnings("ignore")
SETS = {"122 inputs": "avail", "122 + quarterback": "avail+qb", "243 + quarterback": "no_gr+qb"}

if __name__ == "__main__":
    pd.options.display.width = 230
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    tr, te = F[F.season.between(2021, 2025)], F[F.season == 2026].copy()
    sets = feature_sets(F)
    print("inputs:", {k: len(sets[v]) for k, v in SETS.items()}, "| quarterback inputs:", [c for c in sets["avail+qb"] if c.startswith("qbs_")])
    P = te[ID].copy()
    for name, key in SETS.items():
        pr = fit_predict(tr, te, sets[key])
        for c in pr:
            P[f"{c}|{name}"] = pr[c]
        P[f"blend|{name}"] = 0.5 * P.sl + 0.5 * pr["ensemble"]
        print("trained", name, flush=True)
    P.to_pickle("/tmp/ml_qb.pkl")
    P = P.dropna(subset=["sl"])
    pool = P[P.sl >= POOL_MIN]
    print(f"\n=== points accuracy, 2026 weeks 1-9, Sleeper pool n={len(pool)} ===")
    rows = [dict(model="Sleeper (public)", **metrics(pool.pts.to_numpy(float), pool.sl.to_numpy(float)))]
    for name in SETS:
        for kind, lab in (("ensemble", "black box"), ("blend", "50-50 blend")):
            c = f"{kind}|{name}"
            d, lo, hi = vs(pool, c)
            rows.append(dict(model=f"{lab}, {name}", **metrics(pool.pts.to_numpy(float), pool[c].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi))
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    d, lo, hi = vs(pool, "ensemble|122 + quarterback", "ensemble|122 inputs")
    print(f"quarterback inputs: MAE change {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = they help)")
    for pos in MD.POS:
        g = pool[pool.position == pos]
        d, lo, hi = vs(g, "ensemble|122 + quarterback", "ensemble|122 inputs")
        print(f"   {pos}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(g)})")

    rp = P[(P.sl >= POOL_MIN) | (P["ensemble|122 inputs"] >= POOL_MIN)]
    cols = {"Sleeper (public)": "sl", "black box, 122 inputs": "ensemble|122 inputs", "black box, 122 + quarterback": "ensemble|122 + quarterback",
            "black box, 243 + quarterback": "ensemble|243 + quarterback", "50-50 blend, 122 inputs": "blend|122 inputs", "50-50 blend, 122 + quarterback": "blend|122 + quarterback"}
    print(f"\n=== ranking within position and week, pool: either source projects {POOL_MIN:.0f}+ and the player played, n={len(rp)} ===")
    print(evaluate(rp, cols).to_string(index=False, float_format="{:.3f}".format))
    print(by_position(rp, {k: cols[k] for k in ("Sleeper (public)", "black box, 122 + quarterback", "50-50 blend, 122 + quarterback")}).to_string(index=False, float_format="{:.3f}".format))
    for lab, m in (("weeks 1-3", rp.week <= 3), ("weeks 4-9", rp.week >= 4)):
        print(f"\n{lab}:")
        print(evaluate(rp[m], {k: cols[k] for k in ("Sleeper (public)", "black box, 122 + quarterback", "50-50 blend, 122 + quarterback")}, boots=500).to_string(index=False, float_format="{:.3f}".format))
