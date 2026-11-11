"""A distribution per player instead of one number: bust / projected / boom.

projected  the working projection: the black-box ensemble blended 50-50 with Sleeper (the ensemble alone when Sleeper has no number)
bust, boom the 5th and 95th percentiles of the outcome (a 90% interval; `LO`, `HI`), from gradient-boosted quantile regressions

To learn how wide and how skewed outcomes are around a projection, the training seasons need projections made without seeing those games: each of 2021-25 is
projected by an ensemble trained on the other four seasons (cached in .cache/ml_proj.pkl); 2026 by an ensemble trained on 2021-25.
Two versions of the quantile models are compared on 2026:
  simple   quantiles depend on the projection and the position only
  full     they also see every model input, Sleeper's number and how far Sleeper and the ensemble disagree

    uv run python -m analysis.ml_dist          # builds the cache, checks calibration on 2026
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from analysis import ml_data as MD
from analysis.ml_model import POOL_MIN, SL_FEATS, feature_sets, fit_predict

warnings.filterwarnings("ignore")
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
LO, HI = 0.05, 0.95
PROJ = Path(".cache/ml_proj.pkl")
POSF = ["is_QB", "is_RB", "is_WR", "is_TE"]
SIMPLE = ["proj"] + POSF
TRAIN = (2021, 2025)


def projections(F: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Ensemble projection for every row, never from a model that saw that row's season (cached)."""
    if PROJ.exists():
        p = pd.read_pickle(PROJ)
        if len(p) == len(F):
            return p
    parts = []
    seasons = list(range(TRAIN[0], TRAIN[1] + 1))
    for s in seasons:
        tr, te = F[F.season.isin([x for x in seasons if x != s])], F[F.season == s]
        parts.append(te[["player_id", "season", "week"]].assign(ens=fit_predict(tr, te, feats)["ensemble"].to_numpy()))
        print("projected", s, flush=True)
    te = F[F.season > TRAIN[1]]
    parts.append(te[["player_id", "season", "week"]].assign(ens=fit_predict(F[F.season.isin(seasons)], te, feats)["ensemble"].to_numpy()))
    p = pd.concat(parts, ignore_index=True)
    p.to_pickle(PROJ)
    return p


def add_proj(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["proj"] = np.where(X.sl.notna(), 0.5 * X.sl + 0.5 * X.ens, X.ens)
    X["dis"] = X.sl - X.ens                                       # disagreement between the two sources
    return X


def qmodel(q: float, simple: bool):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=400, max_leaf_nodes=6 if simple else 10,
                                         min_samples_leaf=150 if simple else 120, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=30, random_state=0)


def fit_quantiles(tr: pd.DataFrame, cols: list[str], simple: bool) -> dict:
    return {q: qmodel(q, simple).fit(tr[cols], tr.pts) for q in QS}


def predict_quantiles(models: dict, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Quantile predictions, forced to be non-negative and increasing."""
    Q = np.column_stack([models[q].predict(X[cols]) for q in QS])
    Q = np.sort(np.maximum(Q, 0.0), axis=1)
    return pd.DataFrame(Q, index=X.index, columns=[f"q{int(q * 100):02d}" for q in QS])


def knots(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear inverse CDF through the predicted quantiles: 0 at probability 0, and a top end one tail-step beyond the 99th percentile."""
    lo = np.zeros((len(Q), 1))
    hi = Q[:, -1:] + (Q[:, -1:] - Q[:, -2:-1])
    return np.array([0.0] + QS + [1.0]), np.hstack([lo, Q, hi])


def sample(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Outcomes for uniform draws U (n_sims x n_players) given quantile predictions Q (n_players x len(QS))."""
    p, V = knots(Q)
    return np.column_stack([np.interp(U[:, j], p, V[j]) for j in range(len(Q))])


def dist_mean(Q: np.ndarray) -> np.ndarray:
    p, V = knots(Q)
    return ((V[:, 1:] + V[:, :-1]) / 2 * np.diff(p)).sum(axis=1)


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def qcols():
    return [f"q{int(q * 100):02d}" for q in QS]


def full_cols(feats):
    return feats + SL_FEATS + ["proj", "ens", "dis"]


if __name__ == "__main__":
    pd.options.display.width = 220
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    X = add_proj(F.merge(projections(F, feats), on=["player_id", "season", "week"], how="left"))
    tr = X[X.season.between(*TRAIN) & X.sl.notna()]
    te = X[(X.season == 2026) & (X.sl >= POOL_MIN)].copy()
    print(f"quantile models trained on {len(tr)} player-games of {TRAIN[0]}-{TRAIN[1]} (each projected by a model that never saw its season); checked on {len(te)} of 2026")
    print(f"projection accuracy in training rows Sleeper projects at {POOL_MIN:.0f}+: MAE {np.abs(tr[tr.sl >= POOL_MIN].proj - tr[tr.sl >= POOL_MIN].pts).mean():.3f}; 2026: {np.abs(te.proj - te.pts).mean():.3f}")
    res = {}
    for name, cols, simple in (("simple", SIMPLE, True), ("full", full_cols(feats), False)):
        m = fit_quantiles(tr, cols, simple)
        res[name] = predict_quantiles(m, te, cols)
    y = te.pts.to_numpy(float)
    print("\ncalibration on 2026: share of actual scores below each predicted percentile (should match the percentile), and pinball loss (lower is better)")
    rows = []
    for q, c in zip(QS, qcols()):
        rows.append(dict(percentile=int(q * 100), simple_below=(y < res["simple"][c]).mean(), full_below=(y < res["full"][c]).mean(),
                         simple_loss=pinball(y, res["simple"][c].to_numpy(), q), full_loss=pinball(y, res["full"][c].to_numpy(), q)))
    T_ = pd.DataFrame(rows)
    print(T_.to_string(index=False, float_format="{:.3f}".format))
    print(f"total pinball loss: simple {T_.simple_loss.sum():.3f} | full {T_.full_loss.sum():.3f}")
    for name in res:
        Q = res[name]
        inside = ((y >= Q.q05) & (y <= Q.q95)).mean()
        in80 = ((y >= Q.q10) & (y <= Q.q90)).mean()
        dm = dist_mean(Q.to_numpy())
        print(f"{name}: 90% interval covers {inside:.3f} (80% interval {in80:.3f}); average width {np.mean(Q.q95 - Q.q05):.1f}; "
              f"mean of the distribution {dm.mean():.2f} vs projection {te.proj.mean():.2f} vs actual {y.mean():.2f}; corr(width, |miss|) {np.corrcoef(Q.q95 - Q.q05, np.abs(y - te.proj))[0, 1]:.3f}")
    print("\nby position (full): coverage of the 90% interval, and how often a player finished above his boom / below his bust")
    Q = res["full"]
    for pos in MD.POS:
        m = (te.position == pos).to_numpy()
        print(f"  {pos} n={m.sum():4d}: inside {((y[m] >= Q.q05[m]) & (y[m] <= Q.q95[m])).mean():.3f} | above boom {(y[m] > Q.q95[m]).mean():.3f} | below bust {(y[m] < Q.q05[m]).mean():.3f} | "
              f"typical bust / projected / boom: {Q.q05[m].mean():.1f} / {te.proj[m].mean():.1f} / {Q.q95[m].mean():.1f}")
    # does the full model tell players with the same projection apart?
    te["w_full"], te["w_simple"] = (res["full"].q95 - res["full"].q05).to_numpy(), (res["simple"].q95 - res["simple"].q05).to_numpy()
    te["extra"] = te.w_full - te.w_simple
    hi_, lo_ = te[te.extra > te.extra.quantile(0.8)], te[te.extra < te.extra.quantile(0.2)]
    print(f"\nplayers the full model gives a WIDER range than their projection alone implies (top fifth, n={len(hi_)}): actual spread of misses (sd) {np.std(hi_.pts - hi_.proj):.2f}; "
          f"NARROWER (bottom fifth): {np.std(lo_.pts - lo_.proj):.2f}")
    out = te[["player_id", "player_display_name", "position", "week", "proj", "pts"]].join(res["full"])
    out.to_pickle("/tmp/ml_dist_2026.pkl")
    wk = out[out.week == out.week.max()].sort_values("proj", ascending=False).head(12)
    print(f"\nexample, week {int(out.week.max())}: bust / projected / boom and what happened")
    print(wk[["player_display_name", "position", "q05", "proj", "q95", "pts"]].rename(columns={"q05": "bust", "q95": "boom", "pts": "actual"}).to_string(index=False, float_format="{:.1f}".format))
