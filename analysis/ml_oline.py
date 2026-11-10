"""Does offensive-line stability help the 5-season black-box model? Trained 2021-25, tested on 2026.

    uv run python -m analysis.ml_oline
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import ID, POOL_MIN, feature_sets, fit_predict, metrics, vs
from analysis.ml_ranking import evaluate

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 230
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    tr, te = F[F.season.between(2021, 2025)], F[F.season == 2026].copy()
    sets = feature_sets(F)
    ol = [c for c in sets["avail+qb+ol"] if c.startswith("ol_")]
    print("line inputs:", ol, "| present:", F[ol].notna().mean().round(2).to_dict())
    t = F[F.season.between(2021, 2025) & F.position.isin(["RB", "QB"])]
    print("2021-25, same five starters as last game -> average points (RB / QB):",
          {int(k): (round(g[g.position == "RB"].pts.mean(), 2), round(g[g.position == "QB"].pts.mean(), 2), len(g)) for k, g in t.groupby("ol_same5")})
    P = te[ID].copy()
    for name, key in (("base", "avail+qb"), ("line", "avail+qb+ol")):
        pr = fit_predict(tr, te, sets[key])
        for c in pr:
            P[f"{c}|{name}"] = pr[c]
        P[f"blend|{name}"] = 0.5 * P.sl + 0.5 * pr["ensemble"]
        print("trained", name, len(sets[key]), "inputs", flush=True)
    P.to_pickle("/tmp/ml_ol.pkl")
    P = P.dropna(subset=["sl"])
    pool = P[P.sl >= POOL_MIN]
    y = pool.pts.to_numpy(float)
    rows = [dict(model="Sleeper (public)", **metrics(y, pool.sl.to_numpy(float)))]
    for c, lab in (("ensemble|base", "black box, 141 inputs"), ("ensemble|line", "black box + line stability"), ("blend|base", "50-50 blend, 141 inputs"), ("blend|line", "50-50 blend + line stability")):
        d, lo, hi = vs(pool, c)
        rows.append(dict(model=lab, **metrics(y, pool[c].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi))
    print(f"\n2026 weeks 1-9, n={len(pool)}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    d, lo, hi = vs(pool, "ensemble|line", "ensemble|base")
    print(f"line stability: MAE change {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = it helps)")
    for pos in MD.POS:
        g = pool[pool.position == pos]
        d, lo, hi = vs(g, "ensemble|line", "ensemble|base")
        print(f"   {pos}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(g)})")
    for m in ("ridge", "trees", "neural_net"):
        d, lo, hi = vs(pool, f"{m}|line", f"{m}|base")
        print(f"   {m}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rp = P[(P.sl >= POOL_MIN) | (P["ensemble|base"] >= POOL_MIN)]
    print(evaluate(rp, {"Sleeper (public)": "sl", "black box, 141 inputs": "ensemble|base", "black box + line stability": "ensemble|line", "50-50 blend + line stability": "blend|line"}, boots=500
                   ).to_string(index=False, float_format="{:.3f}".format))
