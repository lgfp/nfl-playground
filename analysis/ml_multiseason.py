"""Does more training history help the black-box model? Same models and inputs as analysis/ml_model.py, trained on 1, 3 and 5 previous seasons, tested on 2026.

Each season's inputs still look back one season at most; only the number of training seasons changes.
    uv run python -m analysis.ml_multiseason
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import ID, POOL_MIN, SL_FEATS, feature_sets, fit_predict, gbm, metrics, vs

warnings.filterwarnings("ignore")
TRAIN = {"2025 only": [2025], "2023-25": [2023, 2024, 2025], "2021-25": [2021, 2022, 2023, 2024, 2025]}
SETS = {"122 inputs": "avail", "243 inputs": "no_gr", "288 inputs": "all"}

if __name__ == "__main__":
    pd.options.display.width = 220
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    te = F[F.season == 2026].copy()
    sets = feature_sets(F)
    print("player-games by season:", F.groupby("season").size().to_dict())
    print("share with a Sleeper projection by season:", F.groupby("season").sl.apply(lambda v: round(v.notna().mean(), 2)).to_dict())
    P = te[ID].copy()
    for tname, seasons in TRAIN.items():
        tr = F[F.season.isin(seasons)]
        for sname, key in SETS.items():
            pr = fit_predict(tr, te, sets[key])
            for c in pr:
                P[f"{c}|{sname}|{tname}"] = pr[c]
            P[f"blend|{sname}|{tname}"] = 0.5 * P.sl + 0.5 * pr["ensemble"]
        # stacked: trees that also see Sleeper's projection, trained on the seasons that have one
        trs, tes = tr[tr.sl.notna()], te[te.sl.notna()]
        for sname, key in (("122 inputs", "avail"), ("288 inputs", "all")):
            sf = sets[key] + SL_FEATS
            P.loc[tes.index, f"stacked|{sname}|{tname}"] = np.mean([gbm("squared_error", s).fit(trs[sf], trs.pts).predict(tes[sf]) for s in range(3)], axis=0).clip(min=0)
        only = SL_FEATS + ["is_QB", "is_RB", "is_WR", "is_TE"]
        P.loc[tes.index, f"recal|{tname}"] = np.mean([gbm("squared_error", s).fit(trs[only], trs.pts).predict(tes[only]) for s in range(3)], axis=0).clip(min=0)
        print(f"trained {tname}: {len(tr)} rows ({len(trs)} with a Sleeper projection)", flush=True)
    P.to_pickle("/tmp/ml_multi.pkl")
    pool = P[P.sl >= POOL_MIN].dropna(subset=[c for c in P.columns if "|" in c])
    y = pool.pts.to_numpy(float)

    def row(label, col, ref="sl"):
        m = metrics(y, pool[col].to_numpy(float))
        d, lo, hi = vs(pool, col, ref) if col != ref else (0.0, 0.0, 0.0)
        return dict(model=label, MAE=m["MAE"], RMSE=m["RMSE"], bias=m["bias"], corr=m["corr"], dMAE_vs_Sleeper=d, lo=lo, hi=hi)

    print(f"\n=== 2026 weeks 1-9, players Sleeper projects at {POOL_MIN:.0f}+ who played: n={len(pool)} ===")
    rows = [row("Sleeper (public)", "sl")]
    for kind, lab in (("ensemble", "ENSEMBLE"), ("blend", "50-50 blend with Sleeper"), ("stacked", "trees + Sleeper as input")):
        for sname in SETS:
            for tname in TRAIN:
                c = f"{kind}|{sname}|{tname}"
                if c in pool:
                    rows.append(row(f"{lab}, {sname}, trained {tname}", c))
    for tname in TRAIN:
        rows.append(row(f"Sleeper recalibrated only, trained {tname}", f"recal|{tname}"))
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
    print("\nper model type, 288 inputs: MAE / corr by training span")
    for m in ("ridge", "trees", "trees_poisson", "neural_net", "ensemble"):
        print(f"  {m:14s}", " | ".join(f"{t}: {np.abs(pool[f'{m}|288 inputs|{t}'] - y).mean():.3f} / {np.corrcoef(pool[f'{m}|288 inputs|{t}'], y)[0, 1]:.3f}" for t in TRAIN))
    print("\npaired: 5 seasons minus 1 season of training (MAE, negative = more history helps)")
    for sname in SETS:
        d, lo, hi = vs(pool, f"ensemble|{sname}|2021-25", f"ensemble|{sname}|2025 only")
        print(f"  ensemble, {sname}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    d, lo, hi = vs(pool, "ensemble|288 inputs|2021-25", "ensemble|122 inputs|2021-25")
    print(f"with 5 seasons: 288 inputs minus 122 inputs: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    best = "ensemble|288 inputs|2021-25"
    for lab, m in (("weeks 1-3", pool.week <= 3), ("weeks 4-9", pool.week >= 4)) + tuple((p, pool.position == p) for p in MD.POS):
        d = pool[m]
        x = vs(d, best); b = vs(d, "blend|288 inputs|2021-25")
        print(f"  {lab:9s} n={len(d):4d}: Sleeper {np.abs(d.sl - d.pts).mean():.3f} | ensemble (5 seasons, 288) {np.abs(d[best] - d.pts).mean():.3f} ({x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]) | "
              f"50-50 {np.abs(d['blend|288 inputs|2021-25'] - d.pts).mean():.3f} ({b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}])")
    # what the trees lean on with 5 seasons
    from sklearn.inspection import permutation_importance
    tr = F[F.season.isin(TRAIN["2021-25"])]
    mdl = gbm().fit(tr[sets["all"]], tr.pts)
    imp = pd.Series(permutation_importance(mdl, te[sets["all"]], te.pts, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error").importances_mean, index=sets["all"])
    grp = imp.groupby(lambda c: c.split("_")[0] if c.split("_")[0] in MD.GROUPS else ("availability" if c in MD.AVAIL else "original usage/context")).sum()
    print("\nshare of the trees' reliance by input group, trained on 2021-25:")
    print((grp / grp.sum()).round(3).sort_values(ascending=False).to_string())
    print("top inputs:", imp.sort_values(ascending=False).head(12).round(3).to_dict())
