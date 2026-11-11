"""The "explicit touchdown opportunities" rule, tested: when a projection assumes more TDs than a player's actual scoring chances THIS SEASON justify,
does he fall short?

Explicit chances per game, this season to date (what you would see watching the games): carries inside the 5, carries inside the 10, targets into the end zone,
targets inside the 10. An opportunity-based TD expectation is fitted on earlier seasons only (Poisson: TDs in a game ~ those chances + other carries and
targets + the Vegas implied total). gap = Sleeper's projected TDs - the opportunity-based expectation.

    uv run python -m analysis.ml_td_opps
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_diagnose import trees
from analysis.ml_model import feature_sets
from nflfp import tds as T

warnings.filterwarnings("ignore")
OPP = ["gl_pg", "in10_pg", "ez_pg", "tin10_pg", "car_other_pg", "tgt_other_pg", "imp"]
MIN_GAMES = 3


def table() -> pd.DataFrame:
    d = MD.raw_table()
    d = d[d.position.isin(["RB", "WR", "TE"])].sort_values(["player_id", "season", "week"]).copy()
    d["tds"] = d.rush_td + d.rec_td
    g = d.groupby(["player_id", "season"], sort=False)
    todate = lambda c: g[c].transform(lambda v: v.shift(1).expanding().mean())
    d["gl_pg"], d["ez_pg"] = todate("gl_car"), todate("ez_tgt")
    d["in10_pg"] = todate("car_in10") - d.gl_pg                      # carries from the 6-10
    d["tin10_pg"] = (todate("tgt_in10") - d.ez_pg).clip(lower=0)     # targets inside the 10 that were not thrown into the end zone
    d["car_other_pg"] = todate("carries") - todate("car_in10")
    d["tgt_other_pg"] = (todate("targets") - todate("tgt_in10")).clip(lower=0)
    d["td_pg"] = todate("tds")
    return d[["player_id", "season", "week", "tds", "gl_pg", "in10_pg", "ez_pg", "tin10_pg", "car_other_pg", "tgt_other_pg", "td_pg", "gl_car", "ez_tgt"]]


if __name__ == "__main__":
    pd.options.display.width = 230
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    X = F.merge(table(), on=["player_id", "season", "week"], how="left")
    X["imp"] = (X.implied - 22) / 5
    parts = []
    for ts in (2023, 2024, 2025, 2026):
        tr, te = X[X.season.between(2021, ts - 1)], X[X.season == ts].copy()
        te["model"] = trees(tr, feats)(te)
        te["opp_td"] = np.nan
        for pos in ("RB", "WR", "TE"):                                 # opportunity-based TD expectation, fitted on earlier seasons
            a = tr[(tr.position == pos) & (tr.n_cur >= MIN_GAMES)].dropna(subset=OPP + ["tds"])
            b = (te.position == pos) & (te.n_cur >= MIN_GAMES) & te[OPP].notna().all(axis=1)
            coef = T.poisson_fit(a[OPP].to_numpy(float), a.tds.to_numpy(float), ridge=1.0)
            te.loc[b, "opp_td"] = T.poisson_predict(coef, te.loc[b, OPP].to_numpy(float))
            if ts == 2026:
                print(f"{pos} opportunity model (fitted 2021-25): TD multiplier per extra chance a game: " + ", ".join(f"{n} x{np.exp(c):.2f}" for n, c in zip(OPP, coef[1:])))
        parts.append(te)
    R = pd.concat(parts).dropna(subset=["opp_td", "sl", "sl_tds"])
    R["gap"] = R.sl_tds - R.opp_td
    R["e_td"], R["e_sl"], R["e_m"] = R.sl_tds - R.tds, R.sl - R.pts, R.model - R.pts
    R.to_pickle("/tmp/ml_tdopps.pkl")
    se2 = lambda v: 2 * v.std() / np.sqrt(len(v))

    for pos, floor in (("RB", 8), ("WR", 8), ("TE", 6)):
        d = R[(R.position == pos) & (R.sl >= floor)]
        print(f"\n===== {pos}: Sleeper projects {floor}+, {MIN_GAMES}+ games this season, 2023-26: n={len(d)} =====")
        print(f"TDs per game: actual {d.tds.mean():.3f} | Sleeper projects {d.sl_tds.mean():.3f} | opportunity-based {d.opp_td.mean():.3f}")
        print(f"who predicts the game's TDs better (correlation with actual): Sleeper {np.corrcoef(d.sl_tds, d.tds)[0, 1]:.3f} | opportunity-based {np.corrcoef(d.opp_td, d.tds)[0, 1]:.3f} | "
              f"average of both {np.corrcoef(d.sl_tds + d.opp_td, d.tds)[0, 1]:.3f}")
        c = np.corrcoef(d.gap, d.e_td)[0, 1]
        print(f"does the gap predict Sleeper's TD miss? corr {c:+.3f} (noise band +/-{2 / np.sqrt(len(d)):.3f}); its points miss? corr {np.corrcoef(d.gap, d.e_sl)[0, 1]:+.3f}; "
              f"the model's points miss? corr {np.corrcoef(d.gap, d.e_m)[0, 1]:+.3f}")
        q = pd.qcut(d.gap, 5, labels=["Sleeper far BELOW chances", "2nd", "3rd", "4th", "Sleeper far ABOVE chances"])
        g = d.groupby(q, observed=True).agg(n=("gap", "size"), gap=("gap", "mean"), sleeper_td=("sl_tds", "mean"), opp_td=("opp_td", "mean"), actual_td=("tds", "mean"),
                                            sleeper_td_err=("e_td", "mean"), sleeper_pts_err=("e_sl", "mean"), model_pts_err=("e_m", "mean"))
        print(g.round(2).to_string())
        big = d[d.gap >= 0.25]
        print(f"the rule (Sleeper projects 0.25+ more TDs than his chances imply): n={len(big)}, {big.groupby(['season', 'player_display_name']).ngroups} player-seasons | "
              f"Sleeper TD error {big.e_td.mean():+.2f} +/-{se2(big.e_td):.2f} | Sleeper points error {big.e_sl.mean():+.2f} +/-{se2(big.e_sl):.2f} | model points error {big.e_m.mean():+.2f} +/-{se2(big.e_m):.2f}")
        # a correction: shift Sleeper's projection by k x gap x 6 points, k fitted on earlier seasons
        rows = []
        for ts in (2024, 2025, 2026):
            a, b = d[d.season < ts], d[d.season == ts]
            k = float(np.sum(a.gap * a.e_td) / np.sum(a.gap ** 2))
            rows.append(b.assign(sl_fix=b.sl - 6 * k * b.gap, k=k))
        f = pd.concat(rows)
        diff = np.abs(f.sl_fix - f.pts) - np.abs(f.sl - f.pts)
        print(f"corrected Sleeper (2024-26, n={len(f)}, k per season {f.groupby('season').k.first().round(2).to_dict()}): MAE {np.abs(f.sl - f.pts).mean():.3f} -> {np.abs(f.sl_fix - f.pts).mean():.3f} "
              f"({diff.mean():+.3f} +/-{se2(diff):.3f})")
    b = R[(R.player_display_name == "Chase Brown") & (R.season == 2026)][["week", "gl_pg", "ez_pg", "sl_tds", "opp_td", "gap", "tds", "sl", "model", "pts"]]
    print("\nChase Brown 2026 (chances per game this season going in; projected TDs; actual):")
    print(b.round(2).to_string(index=False))
    w = R[(R.player_display_name == "Kenneth Walker III") & (R.season == 2025)]
    print(f"Kenneth Walker 2025: Sleeper TDs {w.sl_tds.mean():.2f} a game | opportunity-based {w.opp_td.mean():.2f} | actual {w.tds.mean():.2f} | games with gap >= 0.25: {(w.gap >= 0.25).sum()} of {len(w)}")
