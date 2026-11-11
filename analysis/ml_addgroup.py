"""Does adding one group of inputs help the 5-season black-box model? Trained 2021-25, tested on 2026.

    uv run python -m analysis.ml_addgroup <base set> <test set>        e.g.  avail+qb  avail+qb+aly   (set names: analysis.ml_model.feature_sets)
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import ID, POOL_MIN, feature_sets, fit_predict, metrics, vs
from analysis.ml_ranking import evaluate

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pd.options.display.width = 230
    base_key, test_key = sys.argv[1], sys.argv[2]
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    tr, te = F[F.season.between(2021, 2025)], F[F.season == 2026].copy()
    sets = feature_sets(F)
    new = [c for c in sets[test_key] if c not in sets[base_key]]
    print(f"{base_key}: {len(sets[base_key])} inputs; {test_key}: {len(sets[test_key])} inputs; added: {new}")
    P = te[ID].copy()
    for name, key in (("base", base_key), ("test", test_key)):
        pr = fit_predict(tr, te, sets[key])
        for c in pr:
            P[f"{c}|{name}"] = pr[c]
        P[f"blend|{name}"] = 0.5 * P.sl + 0.5 * pr["ensemble"]
        print("trained", name, flush=True)
    P.to_pickle(f"/tmp/ml_add_{test_key.replace('+', '_')}.pkl")
    P = P.dropna(subset=["sl"])
    pool = P[P.sl >= POOL_MIN]
    y = pool.pts.to_numpy(float)
    rows = [dict(model="Sleeper (public)", **metrics(y, pool.sl.to_numpy(float)))]
    for c, lab in (("ensemble|base", "black box, base"), ("ensemble|test", "black box + new inputs"), ("blend|base", "50-50 blend, base"), ("blend|test", "50-50 blend + new inputs")):
        d, lo, hi = vs(pool, c)
        rows.append(dict(model=lab, **metrics(y, pool[c].to_numpy(float)), dMAE_vs_Sleeper=d, lo=lo, hi=hi))
    print(f"\n2026 weeks 1-9, n={len(pool)}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    d, lo, hi = vs(pool, "ensemble|test", "ensemble|base")
    print(f"new inputs: MAE change {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = they help)")
    for pos in MD.POS:
        g = pool[pool.position == pos]
        d, lo, hi = vs(g, "ensemble|test", "ensemble|base")
        print(f"   {pos}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(g)})")
    for m in ("ridge", "trees", "neural_net"):
        d, lo, hi = vs(pool, f"{m}|test", f"{m}|base")
        print(f"   {m}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rp = P[(P.sl >= POOL_MIN) | (P["ensemble|base"] >= POOL_MIN)]
    cols = {"Sleeper (public)": "sl", "black box, base": "ensemble|base", "black box + new inputs": "ensemble|test", "50-50 blend + new inputs": "blend|test"}
    print("\nranking, all positions:")
    print(evaluate(rp, cols, boots=500).to_string(index=False, float_format="{:.3f}".format))
    print("ranking, running backs only:")
    print(evaluate(rp[rp.position == "RB"], cols, boots=500).to_string(index=False, float_format="{:.3f}".format))
