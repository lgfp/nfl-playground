"""Why did the black-box model over-project a player (default: Chase Brown, 2026 week 9), and is his profile over-projected in general?

1. his inputs next to the typical fantasy-relevant RB that week
2. what each family of inputs contributes: swap the family for the typical RB's values and see how the projection moves (trees, trained 2021-25)
3. the archetype test on every RB: are backs whose scoring chances are small for their workload over-projected? (out-of-sample: 2025 from a 2021-24 model, 2026 from 2021-25)

    uv run python -m analysis.ml_diagnose ["Player Name" week]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import feature_sets, gbm

warnings.filterwarnings("ignore")


def trees(tr, feats):
    ms = [gbm("squared_error", s).fit(tr[feats], tr.pts) for s in range(3)]
    return lambda X: np.mean([m.predict(X[feats]) for m in ms], axis=0)


def family(c: str) -> str:
    if c.startswith("qbs_") or c == "qb_new":
        return "quarterback"
    if c.startswith("cal_"):
        return "calendar-weighted form" if "_h" in c else "calendar / gaps"
    if c in ("gap_games", "vac_carry", "vac_target", "n_cur", "n_hist", "n_prev"):
        return "calendar / gaps" if c.startswith("n_") or c == "gap_games" else "teammates out"
    if c.endswith("_prev"):
        return "last season's averages"
    if c in ("implied", "fav", "total_line", "home"):
        return "Vegas lines"
    if c.startswith(("tm_", "def_")):
        return "team volume / defense vs position"
    if c.startswith("is_"):
        return "position"
    stat = c.rsplit("_", 1)[0]
    if stat in ("pts", "pts_notd"):
        return "own recent points"
    if stat in ("rush_td", "rec_td", "pass_td", "xtd_rush", "xtd_rec", "gl_car", "rz_car", "rz_tgt"):
        return "TDs and scoring chances"
    if stat in ("snap_pct", "carry_share", "target_share", "air_share", "carries", "targets", "receptions", "pass_att"):
        return "workload (snaps, carries, targets, shares)"
    return "yards"


if __name__ == "__main__":
    pd.options.display.width = 220
    name = sys.argv[1] if len(sys.argv) > 1 else "Chase Brown"
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(F)["avail+qb+cal"]
    te = F[F.season == 2026]
    f26 = trees(F[F.season.between(2021, 2025)], feats)
    row = te[(te.player_display_name == name) & (te.week == week)]
    peers = te[(te.week == week) & (te.position == row.position.iloc[0]) & (te.sl >= 8)]
    base = float(f26(row)[0])
    print(f"{name}, week {week}: actual {row.pts.iloc[0]:.1f} | trees {base:.1f} | Sleeper {row.sl.iloc[0]:.1f} | typical {row.position.iloc[0]} projected 8+ that week (n={len(peers)}): trees {f26(peers).mean():.1f}")
    show = ["pts_ew", "pts_m8", "pts_cur", "pts_prev", "pts_notd_ew", "cal_pts_h30", "carries_ew", "carries_l1", "carry_share_ew", "targets_ew", "snap_pct_ew", "snap_pct_l1",
            "rush_td_m8", "xtd_rush_m8", "gl_car_m8", "rz_car_m8", "xtd_rec_m8", "implied", "fav", "def_pos_rel", "tm_pts_m5"]
    r = row.iloc[0]
    print("\ninput: his value | median of those peers | his percentile among them")
    for c in show:
        print(f"  {c:16s} {r[c]:7.2f} | {peers[c].median():7.2f} | {(peers[c] < r[c]).mean() * 100:4.0f}%")
    med = peers[feats].median()
    fams = {}
    for c in feats:
        fams.setdefault(family(c), []).append(c)
    print("\nwhat each family adds to his projection, relative to the typical peer (swap the family for the peers' median):")
    out = []
    for fam, cols in fams.items():
        x = row.copy()
        x[cols] = med[cols].to_numpy()
        out.append((fam, base - float(f26(x)[0]), len(cols)))
    for fam, d, n in sorted(out, key=lambda t: -abs(t[1])):
        print(f"  {fam:44s} {d:+5.2f}   ({n} inputs)")
    x = row.copy(); x[feats] = med.to_numpy()
    print(f"  all swapped = the typical peer: {float(f26(x)[0]):.1f}; his projection {base:.1f}")

    # ---- archetype: scoring chances small for the workload
    f25 = trees(F[F.season.between(2021, 2024)], feats)
    R = pd.concat([F[F.season == 2025].assign(pred=lambda d: f25(d)), te.assign(pred=f26(te))])
    R = R[(R.position == "RB") & (R.sl >= 8)].copy()
    opp = R.carries_m8 + R.targets_m8
    R["xtd_per_touch"] = (R.xtd_rush_m8 + R.xtd_rec_m8) / opp.where(opp > 0)
    R["gl_vs_carry"] = R.px_gl_share_ew - R.carry_share_ew                      # share of goal-line carries minus share of all carries
    R["td_minus_xtd"] = R.px_td_over_exp                                        # TDs scored minus expected, last 8
    R["notd_share"] = R.pts_notd_ew / R.pts_ew.where(R.pts_ew > 0)              # how much of his scoring is yards and catches
    R["err"], R["err_sl"] = R.pred - R.pts, R.sl - R.pts
    print(f"\nARCHETYPE TEST: RBs Sleeper projects at 8+, out-of-sample 2025-26, n={len(R)} player-games. error = projection - actual (positive = over-projected)")
    b = R[(R.player_display_name == name) & (R.season == 2026)]
    for col, lab in (("xtd_per_touch", "expected TDs per touch (last 8)"), ("gl_vs_carry", "goal-line share minus carry share"), ("td_minus_xtd", "TDs minus expected TDs (last 8)"),
                     ("notd_share", "share of his points that are not TDs")):
        d = R.dropna(subset=[col])
        q = pd.qcut(d[col], 4, labels=["lowest quarter", "2nd", "3rd", "highest quarter"])
        g = d.groupby(q, observed=True).agg(n=("err", "size"), trees=("err", "mean"), sleeper=("err_sl", "mean"), actual=("pts", "mean"))
        c_t, c_s = np.corrcoef(d[col], d.err)[0, 1], np.corrcoef(d[col], d.err_sl)[0, 1]
        se = 1 / np.sqrt(len(d))
        print(f"\n{lab}: {name} averages {b[col].mean():.3f} (percentile {(d[col] < b[col].mean()).mean() * 100:.0f}%) | corr with over-projection: trees {c_t:+.3f}, Sleeper {c_s:+.3f} (noise band +/-{2 * se:.3f})")
        print(g.round(2).to_string())
    R.to_pickle("/tmp/ml_diag.pkl")
