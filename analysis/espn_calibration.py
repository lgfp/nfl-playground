"""How good are ESPN's weekly projections and their published variances? 2025 weeks 1-18 and 2026 weeks 1-9, against actual points (our scoring), Sleeper and our model.

    uv run python -m analysis.espn_calibration
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis import espn, ml_data as MD, ml_dist as DI
from analysis.ml_model import POOL_MIN
from nflfp import sleeper

warnings.filterwarnings("ignore")
PERIODS = [202500 + w for w in range(1, 19)] + [202600 + w for w in range(1, 10)]


def load() -> pd.DataFrame:
    f = espn.CACHE / "weekly_proj.pkl"
    pl = espn.player_list()
    if not f.exists():
        P = espn.projections([int(i) for i in pl.espn_id], PERIODS)
        P.to_pickle(f)
    P = pd.read_pickle(f)
    return espn.to_points(P).merge(pl[["espn_id", "name", "position"]], on="espn_id")


if __name__ == "__main__":
    pd.options.display.width = 230
    E = load()
    print(f"ESPN weekly projections pulled: {len(E)} player-weeks, {E.espn_id.nunique()} players")
    X, _ = DI.prepare()
    X = X[X.season >= 2025].copy()
    X["key"] = X.player_display_name.map(sleeper.norm) + "|" + X.position
    E["key"] = E.name.map(sleeper.norm) + "|" + E.position
    uniq = X.groupby("key").player_id.nunique()
    E = E[E.key.isin(uniq[uniq == 1].index)]
    D = X.merge(E[["key", "season", "week", "espn_mean", "espn_sd", "espn_td_var_share"]], on=["key", "season", "week"], how="inner")
    D = D[D.sl >= POOL_MIN].dropna(subset=["espn_mean", "proj"])
    D = D[D.espn_mean > 0]
    y = D.pts.to_numpy(float)
    print(f"matched player-games Sleeper projects at {POOL_MIN:.0f}+ with an ESPN projection: {len(D)} ({(D.season == 2025).sum()} in 2025, {(D.season == 2026).sum()} in 2026)")
    print("\npoint accuracy (MAE | bias | corr):")
    for lab, c in (("ESPN", "espn_mean"), ("Sleeper", "sl"), ("our ensemble", "ens"), ("our blend with Sleeper", "proj")):
        for s, g in (("2025", D[D.season == 2025]), ("2026", D[D.season == 2026])):
            e = g[c] - g.pts
            print(f"  {lab:24s} {s}: {np.abs(e).mean():.3f} | {e.mean():+.2f} | {np.corrcoef(g[c], g.pts)[0, 1]:.3f}")
    D["avg3"] = (D.espn_mean + D.sl + D.ens) / 3
    D["blend_espn"] = (D.espn_mean + D.sl + D.ens) / 3
    print("  average of ESPN, Sleeper and our ensemble:", " | ".join(f"{s}: {np.abs(g.avg3 - g.pts).mean():.3f}" for s, g in (("2025", D[D.season == 2025]), ("2026", D[D.season == 2026]))))

    # ESPN's own range, read as a normal distribution around ESPN's mean
    q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    print(f"\nESPN's implied range (mean +/- z x sd of the points; sd average {D.espn_sd.mean():.2f} vs the realized spread of misses {np.std(y - D.espn_mean):.2f})")
    print("  share of actual scores below each ESPN percentile (should equal the percentile):")
    print("  " + " | ".join(f"{int(p * 100)}th: {(y < D.espn_mean + norm.ppf(p) * D.espn_sd).mean():.3f}" for p in q))
    print(f"  ESPN 90% interval covers {((y >= D.espn_mean - 1.645 * D.espn_sd) & (y <= D.espn_mean + 1.645 * D.espn_sd)).mean():.3f}; scaling ESPN's sd by x1.5 covers "
          f"{((y >= D.espn_mean - 1.645 * 1.5 * D.espn_sd) & (y <= D.espn_mean + 1.645 * 1.5 * D.espn_sd)).mean():.3f}; the sd that would give 90%: x{np.quantile(np.abs(y - D.espn_mean) / D.espn_sd, 0.9) / 1.645:.2f}")
    z = (y - D.espn_mean) / D.espn_sd
    print(f"  standardized misses (should have sd 1): sd {z.std():.2f}; share beyond +/-1.645: {(np.abs(z) > 1.645).mean():.3f}")
    # does ESPN's sd rank who is volatile? same projection, different published sd
    D["miss"] = (y - D.espn_mean)
    D["band"] = D.groupby("position").espn_mean.transform(lambda v: pd.qcut(v, 6, labels=False, duplicates="drop"))
    D["sd_rel"] = D.espn_sd / D.groupby(["position", "band"]).espn_sd.transform("median")
    D["miss_rel"] = np.abs(D.miss) / D.groupby(["position", "band"]).miss.transform(lambda v: np.abs(v).mean())
    print(f"\nESPN's published sd relative to others at the same position and projection: 10th-90th percentile {D.sd_rel.quantile(.1):.2f}-{D.sd_rel.quantile(.9):.2f}; "
          f"corr with the relative size of the miss {np.corrcoef(D.sd_rel, D.miss_rel)[0, 1]:+.3f} (noise band +/-{2 / np.sqrt(len(D)):.3f})")
    t = pd.qcut(D.sd_rel, 3, labels=["ESPN says steadiest", "middle", "ESPN says most volatile"])
    print(D.groupby(t, observed=True).agg(n=("pts", "size"), espn_sd=("espn_sd", "mean"), actual_spread_of_misses=("miss", "std"), mean_abs_miss=("miss", lambda v: np.abs(v).mean())).round(2).to_string())
    D["td_sh"] = D.espn_td_var_share
    t2 = pd.qcut(D.espn_td_var_share, 3, labels=["least TD variance", "middle", "most TD variance"])
    print("by the share of ESPN's variance that comes from touchdowns:")
    print(D.groupby(t2, observed=True).agg(n=("pts", "size"), td_share=("espn_td_var_share", "mean"), espn_sd=("espn_sd", "mean"), actual_spread=("miss", "std")).round(2).to_string())
    D.to_pickle("/tmp/espn_cal.pkl")
