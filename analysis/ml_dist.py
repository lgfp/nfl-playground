"""A distribution per player instead of one number: bust / projected / boom.

projected  the working projection: the black-box ensemble blended 50-50 with Sleeper (the ensemble alone when Sleeper has no number)
bust, boom the 5th and 95th percentiles of the outcome (a 90% interval; `LO`, `HI`)

To learn how wide and how skewed outcomes are around a projection, the training seasons need projections made without seeing those games: each of 2021-25 is
projected by an ensemble trained on the other four seasons (cached in .cache/ml_proj.pkl); 2026 by an ensemble trained on 2021-25.

Versions (compared on 2026 by `python -m analysis.ml_dist`):
  simple   gradient-boosted quantile regressions on the projection and the position only: same position + same projection = same range
  medium   the same quantile regressions with the volatility inputs added (`VOL`): the player's own game-to-game spread and biggest game over his last 8, the spread
           of his misses against Sleeper, how much of his scoring and of his projection is touchdowns, target depth, usage mix, snap share, Vegas total, how far
           Sleeper and our ensemble disagree, new quarterback, games missed
  scaled   the simple quantiles, stretched or squeezed around the median by the player's relative volatility: predicted |miss| from the volatility inputs divided
           by predicted |miss| from projection and position alone
  components  built from its parts, so touchdown-heavy projections are wider by construction:
           points = (yards and receptions) + 6 x rushing/receiving TDs + 4 x passing TDs
           TD counts are negative-binomial (Poisson when the data shows no extra spread) around Sleeper's projected TDs, recalibrated to actual counts per position;
           the yards-and-receptions part has its own quantile model around (projection - expected TD points); the two parts move together as they have
           historically (a good game tends to bring both). 4,000 draws per player give his percentiles.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from scipy.stats import nbinom, norm, poisson

from analysis import ml_data as MD
from analysis.ml_model import POOL_MIN, feature_sets, fit_predict
from nflfp import data
from nflfp import tds as T

warnings.filterwarnings("ignore")
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
LO, HI = 0.05, 0.95
PROJ = Path(".cache/ml_proj.pkl")
POSF = ["is_QB", "is_RB", "is_WR", "is_TE"]
SIMPLE = ["proj"] + POSF
VOL = ["pts_sd8", "pts_max8", "miss_sd8", "td_share_hist", "sl_td_share", "dis", "absdis", "px_adot", "gr_tgt_deep_sh", "carries_ew", "targets_ew", "snap_pct_ew",
       "pts_ew", "pts_notd_ew", "xtd_rush_m8", "xtd_rec_m8", "implied", "n_cur", "qb_new", "gap_games"]
MEDIUM = SIMPLE + ["sl", "ens"] + VOL
TRAIN = (2021, 2025)
VERSION = "components"        # used by the matchup simulator: TD variance is explicit. On 2026 it is calibrated (90% interval covers 89.4%) and fits about 0.6%
                              # worse than "simple" / "scaled", which remain available (--version)


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


def volatility(A: pd.DataFrame) -> pd.DataFrame:
    """Per row: spread and maximum of the player's points over his previous 8 games and the spread of his misses against Sleeper, from the current and the
    previous season only. A = every player-game (rows of an upcoming week may be appended with pts = NaN)."""
    A = A.sort_values(["player_id", "season", "week"])
    out = []
    for s in sorted(A.season.unique())[1:]:
        w = A[A.season.isin([s - 1, s])]
        prev = w.groupby("player_id", sort=False).pts.shift(1)
        pmiss = (w.pts - w.sl).groupby(w.player_id, sort=False).shift(1)
        roll = lambda v, f: getattr(v.groupby(w.player_id, sort=False).rolling(8, min_periods=4), f)().reset_index(level=0, drop=True)
        o = w[["player_id", "season", "week"]].assign(pts_sd8=roll(prev, "std"), pts_max8=roll(prev, "max"), miss_sd8=roll(pmiss, "std"))
        out.append(o[o.season == s])
    return pd.concat(out, ignore_index=True)


def add_proj(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["proj"] = np.where(X.sl.notna(), 0.5 * X.sl + 0.5 * X.ens, X.ens)
    X["dis"] = X.sl - X.ens                                       # disagreement between the two sources
    X["absdis"] = X.dis.abs()
    X["td_share_hist"] = 1 - X.pts_notd_ew / X.pts_ew.where(X.pts_ew > 0)
    X["sl_td_share"] = 6 * X.sl_tds / X.sl.where(X.sl > 0)
    return X


def prepare(rows: pd.DataFrame | None = None):
    """(history ready for fitting, upcoming rows ready for prediction). `rows`: pre-game rows with `ensemble` (analysis.ml_week.week_rows)."""
    A = pd.read_pickle(MD.OUT)
    F = A[A.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    keys = ["player_id", "season", "week"]
    base = A[keys + ["pts", "sl"]]
    if rows is not None:
        base = pd.concat([base, rows[keys + ["pts", "sl"]]], ignore_index=True)
    V = volatility(base)
    pw = pd.concat([data.load_player_weeks(y) for y in sorted(F.season.unique())], ignore_index=True)
    cnt = pd.DataFrame({"player_id": pw.player_id, "season": pw.season, "week": pw.week, "td_rr": pw.rushing_tds.fillna(0) + pw.receiving_tds.fillna(0),
                        "td_pass": pw.passing_tds.fillna(0)}).drop_duplicates(keys)
    X = add_proj(F.merge(projections(F, feats), on=keys, how="left").merge(V, on=keys, how="left").merge(cnt, on=keys, how="left"))
    R = None if rows is None else add_proj(rows.assign(ens=rows.ensemble).merge(V, on=keys, how="left"))
    return X, R


def qmodel(q: float, simple: bool):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=400, max_leaf_nodes=6 if simple else 8,
                                         min_samples_leaf=150, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=30, random_state=0)


def qcols():
    return [f"q{int(q * 100):02d}" for q in QS]


def _quantiles(models: dict, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    Q = np.column_stack([models[q].predict(X[cols]) for q in QS])
    return pd.DataFrame(np.sort(np.maximum(Q, 0.0), axis=1), index=X.index, columns=qcols())


def fit_version(tr: pd.DataFrame, version: str):
    """Returns predict(X) -> quantile table (with `rel`, the player's relative volatility, for the scaled version)."""
    if version == "components":
        return fit_components(tr)
    if version in ("simple", "medium"):
        cols = SIMPLE if version == "simple" else MEDIUM
        m = {q: qmodel(q, version == "simple").fit(tr[cols], tr.pts) for q in QS}
        return lambda X: _quantiles(m, X, cols)
    base = {q: qmodel(q, True).fit(tr[SIMPLE], tr.pts) for q in QS}
    y = (tr.pts - tr.proj).abs()
    reg = lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=8, min_samples_leaf=150, l2_regularization=1.0, early_stopping=True,
                                                validation_fraction=0.15, n_iter_no_change=30, random_state=0)
    a, b = reg().fit(tr[SIMPLE], y), reg().fit(tr[MEDIUM], y)

    def predict(X):
        Q = _quantiles(base, X, SIMPLE)
        rel = np.clip(b.predict(X[MEDIUM]) / np.maximum(a.predict(X[SIMPLE]), 0.5), 0.6, 1.7)
        c = Q.q50.to_numpy()[:, None]
        V = np.sort(np.maximum(c + (Q.to_numpy() - c) * rel[:, None], 0.0), axis=1)
        out = pd.DataFrame(V, index=X.index, columns=qcols())
        out["rel"] = rel
        return out
    return predict


def _nb_r(y: np.ndarray, lam: np.ndarray) -> float:
    """Negative-binomial size from the extra spread of counts around their means (inf = Poisson)."""
    extra = float(np.sum((y - lam) ** 2 - lam))
    return float(np.sum(lam ** 2) / extra) if extra > 0 else np.inf


def _count_ppf(u: np.ndarray, lam: np.ndarray, r: float) -> np.ndarray:
    lam = np.maximum(lam, 1e-4)[:, None]
    if not np.isfinite(r) or r > 200:
        return poisson.ppf(u, lam)
    return nbinom.ppf(u, r, r / (r + lam))


def fit_components(tr: pd.DataFrame, draws: int = 4000, seed: int = 0):
    """Distribution from its parts (see the module docstring). Returns predict(X) -> quantile table with td_exp (expected TDs), td_share and sd."""
    par = {}
    for pos in MD.POS:
        d = tr[(tr.position == pos) & tr.td_rr.notna()]
        fit = lambda x, y: T.poisson_fit(np.log(np.maximum(x, 0.02))[:, None], y, ridge=0.01)
        b_rr = fit(d.sl_tds.to_numpy(float), d.td_rr.to_numpy(float))
        lam_rr = T.poisson_predict(b_rr, np.log(np.maximum(d.sl_tds.to_numpy(float), 0.02))[:, None])
        p = dict(b_rr=b_rr, r_rr=_nb_r(d.td_rr.to_numpy(float), lam_rr))
        lam_p = np.zeros(len(d))
        if pos == "QB":
            x = d.sl_pass_tds.fillna(0).to_numpy(float)
            p["b_p"] = fit(x, d.td_pass.to_numpy(float))
            lam_p = T.poisson_predict(p["b_p"], np.log(np.maximum(x, 0.02))[:, None])
            p["r_p"] = _nb_r(d.td_pass.to_numpy(float), lam_p)
        mu_n = np.maximum(d.proj.to_numpy(float) - 6 * lam_rr - 4 * lam_p, 0.2)
        dd = pd.DataFrame({"mu_n": mu_n})
        p["qn"] = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.05, max_iter=300, max_leaf_nodes=6, min_samples_leaf=150,
                                                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=0).fit(dd, d.pts_notd) for q in QS}
        td_pts = 6 * d.td_rr.to_numpy(float) + 4 * d.td_pass.to_numpy(float)
        p["rho"] = float(np.corrcoef(d.pts_notd.to_numpy(float) - mu_n, td_pts - (6 * lam_rr + 4 * lam_p))[0, 1])
        par[pos] = p

    def predict(X: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = []
        for pos in MD.POS:
            d = X[X.position == pos]
            if d.empty:
                continue
            p = par[pos]
            lg = lambda v: np.log(np.maximum(np.nan_to_num(v.to_numpy(float)), 0.02))[:, None]
            lam_rr = T.poisson_predict(p["b_rr"], lg(d.sl_tds))
            lam_p = T.poisson_predict(p["b_p"], lg(d.sl_pass_tds)) if pos == "QB" else np.zeros(len(d))
            no_sl = d.sl.isna().to_numpy()                              # no public projection: assume the position's usual TD share of the projection
            if no_sl.any():
                share = float(np.mean(6 * lam_rr[~no_sl] + 4 * lam_p[~no_sl]) / max(np.mean(d.proj.to_numpy(float)[~no_sl]), 1e-6)) if (~no_sl).any() else 0.25
                lam_rr[no_sl], lam_p[no_sl] = share * d.proj.to_numpy(float)[no_sl] / 6 * (0.0 if pos == "QB" else 1.0), (share * d.proj.to_numpy(float)[no_sl] / 4 if pos == "QB" else 0.0)
            mu_n = np.maximum(d.proj.to_numpy(float) - 6 * lam_rr - 4 * lam_p, 0.2)
            Qn = np.sort(np.maximum(np.column_stack([p["qn"][q].predict(pd.DataFrame({"mu_n": mu_n})) for q in QS]), 0.0), axis=1)
            z = rng.standard_normal((len(d), draws, 3))
            z2 = p["rho"] * z[:, :, 0] + np.sqrt(1 - p["rho"] ** 2) * z[:, :, 1]
            pk, V = knots(Qn)
            n_part = np.vstack([np.interp(norm.cdf(z[j, :, 0]), pk, V[j]) for j in range(len(d))])
            u2 = norm.cdf(z2)
            if pos == "QB":
                tdp = 4 * _count_ppf(u2, lam_p, p["r_p"]) + 6 * _count_ppf(norm.cdf(z[:, :, 2]), lam_rr, p["r_rr"])
            else:
                tdp = 6 * _count_ppf(u2, lam_rr, p["r_rr"])
            tot = n_part + tdp
            q = pd.DataFrame(np.quantile(tot, QS, axis=1).T, index=d.index, columns=qcols())
            q["td_exp"], q["td_share"], q["sd"], q["mean_"] = lam_rr + lam_p, (6 * lam_rr + 4 * lam_p) / np.maximum(d.proj.to_numpy(float), 0.1), tot.std(axis=1), tot.mean(axis=1)
            out.append(q)
        return pd.concat(out).reindex(X.index)                          # rows whose position is not QB/RB/WR/TE get no distribution
    predict.par = par
    return predict


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


if __name__ == "__main__":
    pd.options.display.width = 230
    X, _ = prepare()
    tr = X[X.season.between(*TRAIN) & X.sl.notna()]
    te = X[(X.season == 2026) & (X.sl >= POOL_MIN)].copy()
    y = te.pts.to_numpy(float)
    print(f"trained on {len(tr)} player-games of {TRAIN[0]}-{TRAIN[1]} (each projected by a model that never saw its season); checked on {len(te)} of 2026")
    fits = {v: fit_version(tr, v) for v in ("simple", "scaled", "components")}
    res = {v: f(te) for v, f in fits.items()}
    par = fits["components"].par
    print("components model, per position: TD-count spread (negative-binomial size; inf = Poisson) and how the two parts move together")
    for pos, p in par.items():
        print(f"  {pos}: rush/rec TDs size {p['r_rr']:.1f}" + (f", passing TDs size {p['r_p']:.1f}" if "r_p" in p else "") + f" | correlation of the TD part with the yards-and-receptions part {p['rho']:+.2f}")
    print("\ncalibration on 2026: share of actual scores below each predicted percentile (should equal the percentile) | pinball loss (lower is better)")
    rows = []
    for q, c in zip(QS, qcols()):
        r = dict(percentile=int(q * 100))
        for v in res:
            r[f"{v}_below"] = (y < res[v][c]).mean()
        for v in res:
            r[f"{v}_loss"] = pinball(y, res[v][c].to_numpy(), q)
        rows.append(r)
    T_ = pd.DataFrame(rows)
    print(T_.to_string(index=False, float_format="{:.3f}".format))
    print("total pinball loss:", " | ".join(f"{v} {T_[f'{v}_loss'].sum():.3f}" for v in res))
    rng = np.random.default_rng(0)
    codes = pd.factorize(te.week.astype(str) + te.team)[0]
    for v in ("scaled", "components"):
        d = sum(np.maximum(q * (y - res[v][c]), (q - 1) * (y - res[v][c])) - np.maximum(q * (y - res["simple"][c]), (q - 1) * (y - res["simple"][c])) for q, c in zip(QS, qcols())).to_numpy()
        per = np.bincount(codes, d, codes.max() + 1)
        b = per[rng.integers(0, len(per), size=(2000, len(per)))].sum(1) / len(y)
        print(f"  {v} minus simple, total pinball loss: {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}] (negative = better)")
    for v in res:
        Q = res[v]
        m = Q.mean_.mean() if "mean_" in Q else dist_mean(Q[qcols()].to_numpy()).mean()
        print(f"{v}: 90% interval covers {((y >= Q.q05) & (y <= Q.q95)).mean():.3f} (80%: {((y >= Q.q10) & (y <= Q.q90)).mean():.3f}); average width {np.mean(Q.q95 - Q.q05):.1f}; "
              f"mean of the distribution {m:.2f} vs projection {te.proj.mean():.2f} vs actual {y.mean():.2f}")
    C, S = res["components"], res["simple"]
    te["td_share"], te["w_c"], te["w_s"] = C.td_share.to_numpy(), (C.q95 - C.q05).to_numpy(), (S.q95 - S.q05).to_numpy()
    te["rel_c"] = te.w_c / te.w_s
    te["miss"] = te.pts - te.proj
    te["z"] = te.miss / (te.w_s / 3.29)
    te["boom"], te["bust"] = (y > S.q90).astype(float), (y < S.q10).astype(float)
    print(f"\ncomponents: width of a player's 90% range relative to the simple model (same position and projection): 10th-90th percentile {te.rel_c.quantile(.1):.2f}-{te.rel_c.quantile(.9):.2f}; "
          f"corr with the size of the standardized miss {np.corrcoef(te.rel_c, te.z.abs())[0, 1]:+.3f} (noise band +/-{2 / np.sqrt(len(te)):.3f})")
    for pos in MD.POS:
        d = te[te.position == pos]
        qq = pd.qcut(d.td_share, 3, labels=["low TD share", "middle", "high TD share"])
        g = d.groupby(qq, observed=True).agg(n=("pts", "size"), td_share=("td_share", "mean"), proj=("proj", "mean"), predicted_width=("w_c", "mean"), simple_width=("w_s", "mean"),
                                             actual_spread=("z", "std"), boom=("boom", "mean"), bust=("bust", "mean"))
        print(f"  {pos} by the TD share of the projection:")
        print("   " + g.round(3).to_string().replace("\n", "\n   "))
    out = te[["player_id", "player_display_name", "position", "week", "team", "proj", "pts", "td_share"]].join(C[qcols() + ["td_exp"]])
    out.to_pickle("/tmp/ml_dist_2026.pkl")
    wk = out[out.week == out.week.max()]
    ex = wk[(wk.position == "RB") & wk.proj.between(10, 14)].sort_values("td_share")
    ex = pd.concat([ex.head(4), ex.tail(4)])
    print(f"\nexample, week {int(out.week.max())}: RBs projected 10-14, the four least and four most TD-dependent (components version)")
    print(ex[["player_display_name", "q05", "proj", "q95", "td_exp", "td_share", "pts"]].rename(columns={"q05": "bust", "q95": "boom", "td_exp": "exp TDs", "pts": "actual"}).to_string(
        index=False, formatters={"exp TDs": "{:.2f}".format, "td_share": "{:.2f}".format}, float_format="{:.1f}".format))
