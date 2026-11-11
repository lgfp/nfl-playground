"""Test "scoring context" inputs: does the model need to know HOW a team scores and what share of the scoring chances a player gets?

New inputs (all from earlier games, last 8):
  team: share of expected TDs that are rushing, run rate inside the 20 and inside the 10, rushing TDs per game, rushing share of TDs
  player: his share of the team's expected TDs, expected TDs per touch, goal-line share minus carry share, and Vegas implied total x his expected-TD share
Clean check: train 2021-24, score 2025 (the idea came from a 2026 player, so 2026 is not a fair test of it). Trees only (fast).

    uv run python -m analysis.ml_scoring_context
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_diagnose import trees
from analysis.ml_model import POOL_MIN, feature_sets, vs

warnings.filterwarnings("ignore")


def scoring_context(F: pd.DataFrame) -> pd.DataFrame:
    p = MD.pbp()
    p = p.assign(run=(p.play_type == "run").astype(float), xr=np.where(p.play_type == "run", p.xtd, 0.0), xp=np.where(p.play_type == "pass", p.xtd, 0.0),
                 tdr=np.where(p.play_type == "run", p.td, 0.0), tdp=np.where(p.play_type == "pass", p.td, 0.0))
    g = p.groupby(["season", "week", "posteam"]).agg(xr=("xr", "sum"), xp=("xp", "sum"), tdr=("tdr", "sum"), tdp=("tdp", "sum")).reset_index()
    for lim, nm in ((20, "rz"), (10, "in10")):
        z = p[p.yardline_100 <= lim].groupby(["season", "week", "posteam"]).run.agg(["sum", "size"]).reset_index().rename(columns={"sum": f"{nm}_run", "size": f"{nm}_n"})
        g = g.merge(z, on=["season", "week", "posteam"], how="left")
    g = g.fillna(0.0).sort_values(["posteam", "season", "week"])
    r8 = lambda c: g.groupby("posteam")[c].shift(1).groupby(g.posteam).rolling(8, min_periods=2).sum().reset_index(level=0, drop=True)
    n8 = g.groupby("posteam").xr.shift(1).groupby(g.posteam).rolling(8, min_periods=2).count().reset_index(level=0, drop=True)
    t = g[["season", "week", "posteam"]].rename(columns={"posteam": "team"}).copy()
    xr, xp, tdr, tdp = r8("xr"), r8("xp"), r8("tdr"), r8("tdp")
    t["sc_tm_rush_xtd_share"] = xr / (xr + xp)
    t["sc_tm_xtd_pg"] = (xr + xp) / n8
    t["sc_tm_rush_td_pg"] = tdr / n8
    t["sc_tm_rush_td_share"] = tdr / (tdr + tdp).where((tdr + tdp) > 0)
    t["sc_tm_rz_run_rate"] = r8("rz_run") / r8("rz_n").where(r8("rz_n") > 0)
    t["sc_tm_in10_run_rate"] = r8("in10_run") / r8("in10_n").where(r8("in10_n") > 0)
    X = F.merge(t, on=["season", "week", "team"], how="left")
    own = X.xtd_rush_m8 + X.xtd_rec_m8
    touches = X.carries_m8 + X.targets_m8
    X["sc_xtd_share"] = own / X.sc_tm_xtd_pg.where(X.sc_tm_xtd_pg > 0)
    X["sc_xtd_per_touch"] = own / touches.where(touches > 0)
    X["sc_gl_minus_carry"] = X.px_gl_share_ew - X.carry_share_ew
    X["sc_implied_x_share"] = X.implied * X.sc_xtd_share
    X["sc_td_minus_xtd"] = X.px_td_over_exp
    return X


if __name__ == "__main__":
    pd.options.display.width = 220
    F = pd.read_pickle(MD.OUT)
    F = F[F.n_hist >= 1].reset_index(drop=True)
    base = feature_sets(F)["avail+qb+cal"]
    F = scoring_context(F)
    sc = [c for c in F.columns if c.startswith("sc_")]
    print("new inputs:", sc)
    r = F[(F.player_display_name == "Chase Brown") & (F.season == 2026) & (F.week == 9)].iloc[0]
    peers = F[(F.season == 2026) & (F.week == 9) & (F.position == "RB") & (F.sl >= 8)]
    print("Chase Brown week 9 | median RB projected 8+ | his percentile")
    for c in sc:
        print(f"  {c:22s} {r[c]:6.3f} | {peers[c].median():6.3f} | {(peers[c] < r[c]).mean() * 100:3.0f}%")
    for label, tr_s, te_s in (("CLEAN CHECK: trained 2021-24, scored on 2025", (2021, 2024), 2025), ("trained 2021-25, scored on 2026 weeks 1-9", (2021, 2025), 2026)):
        tr, te = F[F.season.between(*tr_s)], F[F.season == te_s].copy()
        te["base"], te["new"] = trees(tr, base)(te), trees(tr, base + sc)(te)
        pool = te[te.sl >= POOL_MIN]
        print(f"\n{label}: n={len(pool)}")
        for lab, m in (("all positions", pool.position.notna()), ("RB", pool.position == "RB"), ("WR", pool.position == "WR"), ("TE", pool.position == "TE"), ("QB", pool.position == "QB")):
            d = pool[m]
            x = vs(d, "new", "base")
            print(f"  {lab:14s} n={len(d):4d}: MAE {np.abs(d.base - d.pts).mean():.3f} -> {np.abs(d.new - d.pts).mean():.3f}  ({x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]) | Sleeper {np.abs(d.sl - d.pts).mean():.3f}"
                  f" | corr {np.corrcoef(d.base, d.pts)[0, 1]:.3f} -> {np.corrcoef(d.new, d.pts)[0, 1]:.3f}")
        if te_s == 2026:
            b = te[te.player_display_name == "Chase Brown"][["week", "pts", "base", "new", "sl"]]
            print("  Chase Brown by week:"); print(b.round(1).to_string(index=False))
            print(f"  his mean error: base {np.mean(b.base - b.pts):+.2f} | with scoring context {np.mean(b.new - b.pts):+.2f} | Sleeper {np.mean(b.sl - b.pts):+.2f}")
            rb = pool[pool.position == "RB"]
            low = rb[rb.sc_xtd_share < rb.sc_xtd_share.quantile(0.25)]
            print(f"  RBs in the lowest quarter of expected-TD share (n={len(low)}): mean error base {np.mean(low.base - low.pts):+.2f} | new {np.mean(low.new - low.pts):+.2f} | Sleeper {np.mean(low.sl - low.pts):+.2f}")
