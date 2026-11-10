"""Black-box player model: learn fantasy points per player-game from raw inputs (analysis/ml_data.py), trained on the previous season, tested on 2026.

Models: ridge regression (one per position), gradient-boosted trees (squared and Poisson loss), a small neural net (average of 5 seeds), and their average.
Compared on the same player-games with Sleeper's pre-game projections, and blended with them: the user's schedule (week 1 = 100% public ... week 9+ = 0%),
fixed weights, and a weight learned on 2025. "Stacked" = the trees also get Sleeper's projection as an input (does our data add anything to it?).

    uv run python -m analysis.ml_model
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import ml_data as MD

warnings.filterwarnings("ignore")
ID = ["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts", "pts_notd", "sl", "sl_carries", "sl_targets", "sl_attempts"]
SL_FEATS = ["sl", "sl_carries", "sl_targets", "sl_attempts"]
POOL_MIN = 4.0


def feature_sets(F: pd.DataFrame) -> dict[str, list[str]]:
    extra = [c for c in F.columns if c.split("_")[0] in MD.GROUPS]
    base = [c for c in F.columns if c not in ID and c not in MD.AVAIL and c not in extra]
    return {"base": base, "avail": base + MD.AVAIL, "all": base + MD.AVAIL + extra, "no_gr": base + MD.AVAIL + [c for c in extra if not c.startswith("gr_")]}


def gbm(loss="squared_error", seed=0):
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.04, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=seed)


def linear():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 13)))


def net(seed):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=0.002, max_iter=400, early_stopping=True, validation_fraction=0.15,
                                      n_iter_no_change=20, random_state=seed))


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Fit every model on `tr`, predict `te`. Returns one column per model."""
    out = pd.DataFrame(index=te.index)
    Xtr, ytr, Xte = tr[feats], tr.pts.to_numpy(float), te[feats]
    r = pd.Series(np.nan, index=te.index)
    for pos in MD.POS:                                   # ridge: one model per position
        a, b = tr.position == pos, te.position == pos
        if b.any():
            r[b] = linear().fit(Xtr[a], ytr[a]).predict(Xte[b])
    out["ridge"] = r.clip(lower=0)
    out["trees"] = np.mean([gbm("squared_error", s).fit(Xtr, ytr).predict(Xte) for s in range(3)], axis=0).clip(min=0)
    out["trees_poisson"] = np.mean([gbm("poisson", s).fit(Xtr, np.maximum(ytr, 0)).predict(Xte) for s in range(3)], axis=0)
    out["neural_net"] = np.mean([net(s).fit(Xtr, ytr).predict(Xte) for s in range(5)], axis=0).clip(min=0)
    out["ensemble"] = out[["ridge", "trees", "trees_poisson", "neural_net"]].mean(axis=1)
    return out


def oof(tr: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Out-of-fold predictions on the training season (5 folds of whole weeks), for learning blend weights."""
    parts = []
    for a, b in GroupKFold(5).split(tr, groups=tr.week):
        parts.append(fit_predict(tr.iloc[a], tr.iloc[b], feats))
    return pd.concat(parts).loc[tr.index]


def metrics(y, p):
    e = p - y
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), corr=np.corrcoef(p, y)[0, 1], R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def vs(d: pd.DataFrame, col: str, ref: str = "sl", n=2000, seed=0):
    """Paired difference in absolute error (col - ref), bootstrap over team-weeks. Negative = col better."""
    x = (d[col] - d.pts).abs() - (d[ref] - d.pts).abs()
    codes = pd.factorize(d.season.astype(str) + "-" + d.week.astype(str) + "-" + d.team)[0]
    per = np.bincount(codes, x.to_numpy(), codes.max() + 1)
    rng = np.random.default_rng(seed)
    b = per[rng.integers(0, len(per), size=(n, len(per)))].sum(1) / len(x)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def table(d: pd.DataFrame, cols: dict, label: str):
    rows = []
    for name, c in cols.items():
        m = metrics(d.pts.to_numpy(float), d[c].to_numpy(float))
        diff, lo, hi = vs(d, c) if c != "sl" else (0.0, 0.0, 0.0)
        rows.append(dict(model=name, **m, dMAE_vs_Sleeper=diff, lo=lo, hi=hi))
    print(f"\n--- {label}: n={len(d)}, mean actual {d.pts.mean():.2f} ---")
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    pd.options.display.width = 220
    F = pd.read_pickle(MD.OUT) if MD.OUT.exists() else MD.build()
    F = F[F.n_hist >= 1].reset_index(drop=True)                    # at least one earlier game since last season
    tr, te = F[F.season == 2025].copy(), F[F.season == 2026].copy()
    sets = feature_sets(F)
    print(f"train: {len(tr)} player-games of 2025; test: {len(te)} of 2026 (weeks {te.week.min()}-{te.week.max()}); inputs: base {len(sets['base'])}, with availability {len(sets['avail'])}, with everything {len(sets['all'])}")

    P = te[ID].copy()
    for tag, feats in sets.items():
        pr = fit_predict(tr, te, feats)
        for c in pr:
            P[f"{c}|{tag}"] = pr[c]
    # walk-forward: also learn from this season's earlier weeks (refit each week)
    wf = pd.Series(np.nan, index=te.index)
    for w in sorted(te.week.unique()):
        cur = pd.concat([tr, te[te.week < w]])
        b = te.week == w
        wf[b] = np.mean([gbm("squared_error", s).fit(cur[sets["avail"]], cur.pts).predict(te.loc[b, sets["avail"]]) for s in range(3)], axis=0).clip(min=0)
    P["trees|avail|walk-forward"] = wf

    # ---- blends with Sleeper. Learn weights on 2025 out-of-fold predictions (pool: Sleeper >= 4)
    best = "ensemble|avail"
    o = oof(tr, sets["avail"])
    t2 = tr.assign(ml=o["ensemble"])
    t2 = t2[t2.sl >= POOL_MIN]
    A_ = np.column_stack([t2.ml, t2.sl])
    w_ml = float(np.clip(np.linalg.lstsq(A_, t2.pts.to_numpy(float), rcond=None)[0][0] / np.linalg.lstsq(A_, t2.pts.to_numpy(float), rcond=None)[0].sum(), 0, 1))
    grid = {w: np.abs(w * t2.ml + (1 - w) * t2.sl - t2.pts).mean() for w in np.round(np.arange(0, 1.01, 0.1), 1)}
    w_mae = min(grid, key=grid.get)
    print(f"\n2025 (out-of-fold, n={len(t2)}): MAE Sleeper {np.abs(t2.sl - t2.pts).mean():.3f} | ensemble {np.abs(t2.ml - t2.pts).mean():.3f} | "
          f"least-squares weight on our model {w_ml:.2f} | MAE-best weight {w_mae:.1f} (MAE {grid[w_mae]:.3f})")
    print("   2025 MAE by weight on our model:", {k: round(v, 3) for k, v in grid.items()})
    o_all = oof(tr, sets["all"])
    t3 = tr.assign(ml=o["ensemble"], ml_all=o_all["ensemble"])
    t3 = t3[t3.sl >= POOL_MIN]
    print(f"2025 out-of-fold, same games: MAE with {len(sets['avail'])} inputs {np.abs(t3.ml - t3.pts).mean():.3f} | with all {len(sets['all'])} inputs {np.abs(t3.ml_all - t3.pts).mean():.3f} | "
          f"corr {np.corrcoef(t3.ml, t3.pts)[0, 1]:.3f} vs {np.corrcoef(t3.ml_all, t3.pts)[0, 1]:.3f}")
    P["blend|50-50 all"] = 0.5 * P.sl + 0.5 * P["ensemble|all"]
    pub = np.clip((9 - P.week) / 8, 0, 1)                           # the user's schedule: week 1 all public, week 9 on none
    P["blend|schedule"] = pub * P.sl + (1 - pub) * P[best]
    P["blend|50-50"] = 0.5 * P.sl + 0.5 * P[best]
    P["blend|learned"] = (1 - w_ml) * P.sl + w_ml * P[best]
    # ---- stacked: trees that also see Sleeper's projection (train on 2025 rows that have one)
    sf = sets["avail"] + SL_FEATS
    trs = tr[tr.sl.notna()]
    tes = te[te.sl.notna()]
    P.loc[tes.index, "stacked|trees+Sleeper"] = np.mean([gbm("squared_error", s).fit(trs[sf], trs.pts).predict(tes[sf]) for s in range(3)], axis=0).clip(min=0)
    P.loc[tes.index, "stacked|Sleeper recalibrated"] = np.mean([gbm("squared_error", s).fit(trs[SL_FEATS + ["is_QB", "is_RB", "is_WR", "is_TE"]], trs.pts).predict(
        tes[SL_FEATS + ["is_QB", "is_RB", "is_WR", "is_TE"]]) for s in range(3)], axis=0).clip(min=0)
    sfa = sets["all"] + SL_FEATS
    P.loc[tes.index, "stacked|all+Sleeper"] = np.mean([gbm("squared_error", s).fit(trs[sfa], trs.pts).predict(tes[sfa]) for s in range(3)], axis=0).clip(min=0)
    P.to_pickle("/tmp/ml_preds.pkl")

    pool = P[(P.sl >= POOL_MIN)].dropna(subset=[best, "stacked|trees+Sleeper"])
    cols = {"Sleeper (public)": "sl", "ridge": "ridge|avail", "trees": "trees|avail", "trees, Poisson": "trees_poisson|avail", "neural net": "neural_net|avail",
            "ENSEMBLE (trained on 2025)": "ensemble|avail", "ensemble, no availability inputs": "ensemble|base",
            "ENSEMBLE, ALL INPUTS": "ensemble|all", "  ridge, all inputs": "ridge|all", "  trees, all inputs": "trees|all", "  neural net, all inputs": "neural_net|all",
            "blend: 50-50 with all inputs": "blend|50-50 all", "stacked: trees + all inputs + Sleeper": "stacked|all+Sleeper", "trees, refit every week (walk-forward)": "trees|avail|walk-forward",
            "blend: your schedule (wk1 public -> wk9 ours)": "blend|schedule", "blend: 50-50": "blend|50-50", f"blend: weight learned on 2025 ({w_ml:.2f} ours)": "blend|learned",
            "stacked: trees + Sleeper as input": "stacked|trees+Sleeper", "Sleeper recalibrated only": "stacked|Sleeper recalibrated"}
    table(pool, cols, f"2026 weeks 1-9, players Sleeper projects at {POOL_MIN:.0f}+ who played")
    table(pool[pool.week <= 3], cols, "weeks 1-3")
    table(pool[pool.week >= 4], cols, "weeks 4-9")
    x = vs(pool, "ensemble|all", "ensemble|avail")
    print(f"\nall inputs minus the earlier {len(sets['avail'])} inputs, MAE on 2026: {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}] (negative = more inputs help)")
    x = vs(pool, "ensemble|all", "ensemble|no_gr")
    print(f"with situational detail ({len(sets['all'])} inputs) minus without it ({len(sets['no_gr'])}), MAE on 2026: {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}] | "
          f"MAE {np.abs(pool['ensemble|all'] - pool.pts).mean():.3f} vs {np.abs(pool['ensemble|no_gr'] - pool.pts).mean():.3f}; "
          f"corr {np.corrcoef(pool['ensemble|all'], pool.pts)[0, 1]:.3f} vs {np.corrcoef(pool['ensemble|no_gr'], pool.pts)[0, 1]:.3f}")
    short = {k: cols[k] for k in ("Sleeper (public)", "ENSEMBLE (trained on 2025)", "ENSEMBLE, ALL INPUTS", "blend: your schedule (wk1 public -> wk9 ours)", "blend: 50-50", "stacked: trees + Sleeper as input")}
    for pos in MD.POS:
        table(pool[pool.position == pos], short, f"{pos}, weeks 1-9")
    try:                                                               # the hand-built modules 1-4 on the weeks we have them for (4-8)
        H = pd.read_pickle("/tmp/backtest_players.pkl")[["pid", "week", "ours"]].rename(columns={"pid": "player_id", "ours": "modules"}).dropna().drop_duplicates(["player_id", "week"])
        h = pool.merge(H, on=["player_id", "week"])
        table(h, {"Sleeper (public)": "sl", "hand-built modules 1-4": "modules", "ENSEMBLE (trained on 2025)": "ensemble|avail", "blend: 50-50": "blend|50-50",
                  "stacked: trees + Sleeper as input": "stacked|trees+Sleeper"}, "weeks 4-8, same player-games as the modules 1-4 backtest")
    except FileNotFoundError:
        pass
    # what the trees lean on (permutation importance on 2026, squared error)
    from sklearn.inspection import permutation_importance
    m = gbm().fit(tr[sets["all"]], tr.pts)
    imp = permutation_importance(m, te[sets["all"]], te.pts, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.Series(imp.importances_mean, index=sets["all"])
    grp = imp.groupby(lambda c: c.split("_")[0] if c.split("_")[0] in MD.GROUPS else ("availability" if c in MD.AVAIL else "original usage/context")).sum()
    print("\nshare of the trees' reliance by input group (sum of permutation importance):")
    print((grp / grp.sum()).round(3).sort_values(ascending=False).to_string())
    top = imp.sort_values(ascending=False).head(15)
    print("\ninputs the trees lean on most (rise in MAE when the input is shuffled, 2026):")
    print(top.round(3).to_string())
