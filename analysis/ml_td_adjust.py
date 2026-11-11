"""A touchdown adjustment for running backs, tested inside the blend.

The rule: when the public projection assumes more TDs than a back's scoring chances justify, mark him down (and up in the opposite case).
  opp_td   opportunity-based TDs per game: Poisson fit on earlier seasons of TDs ~ this season's carries inside the 5 / 6-10, end-zone targets, other targets
           inside the 10, other carries and targets (all per game, season to date) + the Vegas implied total      (analysis/ml_td_opps.py)
  gap      Sleeper's projected rushing + receiving TDs minus opp_td
  adjusted projection = projection - k x gap, with k fitted on earlier seasons to that projection's own points error
Needs 3+ games this season; earlier weeks are left alone. Everything is out-of-sample: season t uses fits from seasons before t.

    uv run python -m analysis.ml_td_adjust
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_diagnose import trees
from analysis.ml_model import POOL_MIN, feature_sets, fit_predict, vs
from analysis.ml_ranking import evaluate
from analysis.ml_td_opps import MIN_GAMES, OPP, table
from nflfp import tds as T

warnings.filterwarnings("ignore")


def add_opportunity_tds(X: pd.DataFrame, seasons) -> pd.DataFrame:
    """Adds `opp_td` for RB rows of each season in `seasons`, from a Poisson fit on the seasons before it."""
    X = X.copy()
    X["opp_td"] = np.nan
    for ts in seasons:
        a = X[(X.season < ts) & (X.season >= 2021) & (X.position == "RB") & (X.n_cur >= MIN_GAMES)].dropna(subset=OPP + ["tds"])
        b = (X.season == ts) & (X.position == "RB") & (X.n_cur >= MIN_GAMES) & X[OPP].notna().all(axis=1)
        coef = T.poisson_fit(a[OPP].to_numpy(float), a.tds.to_numpy(float), ridge=1.0)
        X.loc[b, "opp_td"] = T.poisson_predict(coef, X.loc[b, OPP].to_numpy(float))
    X["gap"] = X.sl_tds - X.opp_td
    return X


def fit_k(d: pd.DataFrame, col: str) -> float:
    """Points to take off per TD of gap: least squares through the origin of (projection - actual) on the gap."""
    d = d.dropna(subset=["gap", col])
    return float(np.sum(d.gap * (d[col] - d.pts)) / np.sum(d.gap ** 2))


if __name__ == "__main__":
    pd.options.display.width = 230
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    X = F.merge(table(), on=["player_id", "season", "week"], how="left")
    X["imp"] = (X.implied - 22) / 5
    X = add_opportunity_tds(X, (2023, 2024, 2025, 2026))
    X["model"] = np.nan
    for ts in (2023, 2024, 2025, 2026):                               # tree model from earlier seasons only (the quick stand-in for the ensemble)
        m = X.season == ts
        X.loc[m, "model"] = trees(X[X.season.between(2021, ts - 1)], feats)(X[m])
    te26 = X[X.season == 2026]
    X.loc[te26.index, "ens"] = fit_predict(X[X.season.between(2021, 2025)], te26, feats)["ensemble"]      # the full ensemble for 2026
    X["blend"] = 0.5 * X.sl + 0.5 * X.model
    X["blend_ens"] = 0.5 * X.sl + 0.5 * X.ens
    rb = X[(X.position == "RB") & (X.sl >= POOL_MIN)].copy()
    out = []
    for ts in (2024, 2025, 2026):
        past, cur = rb[(rb.season < ts) & (rb.season >= 2023)], rb[rb.season == ts].copy()
        k_sl, k_bl, k_md = fit_k(past, "sl"), fit_k(past, "blend"), fit_k(past, "model")
        g = cur.gap.fillna(0.0)                                       # fewer than 3 games this season: no adjustment
        cur["sl_adj"] = cur.sl - k_sl * g
        cur["blend_adj"] = cur.blend - k_bl * g
        cur["blend_sl_adj"] = 0.5 * cur.sl_adj + 0.5 * cur.model
        cur["blend_ens_adj"] = cur.blend_ens - k_bl * g
        cur["k_sl"], cur["k_bl"], cur["k_md"] = k_sl, k_bl, k_md
        out.append(cur)
    R = pd.concat(out)
    R.to_pickle("/tmp/ml_td_adjust.pkl")
    print("points taken off per projected TD above his chances, fitted on earlier seasons:")
    print(R.groupby("season")[["k_sl", "k_bl", "k_md"]].first().rename(columns={"k_sl": "Sleeper", "k_bl": "blend", "k_md": "model alone"}).round(2).to_string())

    def report(d, cols, label):
        print(f"\n{label}: n={len(d)} RB games ({d.gap.notna().sum()} with an adjustment)")
        base = {"sl_adj": "sl", "blend_adj": "blend", "blend_sl_adj": "blend", "blend_ens_adj": "blend_ens"}
        for name, c in cols.items():
            line = f"  {name:44s} MAE {np.abs(d[c] - d.pts).mean():.3f} | corr {np.corrcoef(d[c], d.pts)[0, 1]:.3f}"
            if c in base:
                x = vs(d, c, base[c])
                line += f" | change vs unadjusted {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
            print(line)

    cols = {"Sleeper": "sl", "Sleeper, TD-adjusted": "sl_adj", "blend (Sleeper + trees)": "blend", "blend, TD-adjusted (k fitted to the blend)": "blend_adj",
            "blend using TD-adjusted Sleeper": "blend_sl_adj"}
    report(R, cols, "2024-26, RBs Sleeper projects 4+")
    report(R[R.gap.notna()], cols, "2024-26, only games with an adjustment (3+ games this season)")
    for s in (2024, 2025, 2026):
        d = R[(R.season == s) & R.gap.notna()]
        x, y = vs(d, "sl_adj", "sl"), vs(d, "blend_adj", "blend")
        print(f"  {s}: n={len(d)} | Sleeper {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}] | blend {y[0]:+.3f} [{y[1]:+.3f}, {y[2]:+.3f}]")
    big = R[R.gap.abs() >= 0.25]
    report(big, cols, "2024-26, games where the gap is 0.25+ TDs either way")
    r26 = R[R.season == 2026]
    cols26 = {"Sleeper": "sl", "Sleeper, TD-adjusted": "sl_adj", "blend (Sleeper + full ensemble)": "blend_ens", "blend with the full ensemble, TD-adjusted": "blend_ens_adj"}
    report(r26, cols26, "2026 weeks 1-9 with the full ensemble")
    report(r26[r26.gap.notna()], cols26, "2026 weeks 4-9 (adjusted games only)")
    print("\nranking of RBs within a week (2024-26):")
    e = evaluate(R, {"Sleeper": "sl", "Sleeper, TD-adjusted": "sl_adj", "blend": "blend", "blend, TD-adjusted": "blend_adj"}, boots=400)
    print(e[["model", "spearman", "pairs", "top_N", "d_pairs_vs_Sleeper", "lo", "hi"]].to_string(index=False, float_format="{:.3f}".format))
    # all positions in 2026: what the RB-only change does to the headline number
    P = X[(X.season == 2026) & (X.sl >= POOL_MIN)].dropna(subset=["blend_ens"]).copy()
    P = P.merge(r26[["player_id", "week", "blend_ens_adj"]], on=["player_id", "week"], how="left")
    P["final"] = P.blend_ens_adj.fillna(P.blend_ens)
    x, y = vs(P, "final", "blend_ens"), vs(P, "final", "sl")
    print(f"\nall positions, 2026 weeks 1-9 (n={len(P)}): Sleeper {np.abs(P.sl - P.pts).mean():.3f} | blend {np.abs(P.blend_ens - P.pts).mean():.3f} | blend with the RB TD adjustment "
          f"{np.abs(P.final - P.pts).mean():.3f} (vs blend {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]; vs Sleeper {y[0]:+.3f} [{y[1]:+.3f}, {y[2]:+.3f}])")
    b = r26[r26.player_display_name == "Chase Brown"][["week", "sl_tds", "opp_td", "gap", "sl", "sl_adj", "blend_ens", "blend_ens_adj", "pts"]]
    print("\nChase Brown 2026:")
    print(b.round(2).to_string(index=False))
    moved = r26.dropna(subset=["gap"]).assign(move=lambda d: d.blend_ens_adj - d.blend_ens)
    print("\nbiggest moves in 2026 (blend, points):")
    print(moved.reindex(moved.move.abs().sort_values(ascending=False).index).head(10)[["player_display_name", "week", "sl_tds", "opp_td", "gap", "move", "blend_ens", "pts"]].round(2).to_string(index=False))
