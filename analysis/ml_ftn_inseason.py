"""Does FTN's in-season DVOA (as it stood before each game) explain what our blend and Sleeper miss? 2026 only, since that is the only season of
weekly FTN data available (analysis/ftn.weekly_games, rebuilt to-date from single-game DVOA; caveat: single-game numbers use FTN's opponent adjustments
as of the export, a mild look-ahead).

A small correction is learned on the misses (actual - projection) from to-date DVOA of the player's offense and the opposing defense, one ridge per
position, and scored two ways: leave-one-week-out (uses later weeks too: is there signal at all?) and forward only (weeks before the one predicted).

    uv run python -m analysis.ml_ftn_inseason [predictions pickle from analysis.ml_ftn, default /tmp/ml_ftn.pkl]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import ftn
from analysis.ml_model import POOL_MIN, vs

FEATS = ["o_def", "o_def_pass", "o_def_rush", "t_off", "t_off_pass", "t_off_rush"]
MIN_GAMES = 2


def correction(d: pd.DataFrame, base: str, train_mask, test_mask) -> pd.Series:
    out = pd.Series(np.nan, index=d.index)
    for pos, g in d.groupby("position"):
        tr, te = g[train_mask.loc[g.index]], g[test_mask.loc[g.index]]
        if len(tr) < 50 or te.empty:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=50.0)).fit(tr[FEATS], tr.pts - tr[base])
        out[te.index] = m.predict(te[FEATS])
    return out


if __name__ == "__main__":
    P = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_ftn.pkl").dropna(subset=["sl"])
    P = P[P.sl >= POOL_MIN].copy()
    td = ftn.to_date(ftn.weekly_games(2026))
    own = td.rename(columns={c: f"t_{c}" for c in td.columns if c not in ("season", "week", "team")})
    opp = td.rename(columns={"team": "opp", **{c: f"o_{c}" for c in td.columns if c not in ("season", "week", "team")}})
    d = P.merge(own, on=["season", "week", "team"], how="left").merge(opp, on=["season", "week", "opp"], how="left")
    d = d[(d.t_n_games >= MIN_GAMES) & (d.o_n_games >= MIN_GAMES)].reset_index(drop=True)
    print(f"2026 weeks {d.week.min()}-{d.week.max()}, {len(d)} player-games with >= {MIN_GAMES} games of FTN data for both teams")
    print("raw correlation of each to-date DVOA with the miss of the blend, by position (pass D for QB/WR/TE, rush D for RB):")
    for pos, g in d.groupby("position"):
        r = g.pts - g["blend|base"]
        print(f"   {pos}: opp pass D {np.corrcoef(g.o_def_pass, r)[0, 1]:+.3f}  opp rush D {np.corrcoef(g.o_def_rush, r)[0, 1]:+.3f}  "
              f"own pass O {np.corrcoef(g.t_off_pass, r)[0, 1]:+.3f}  own rush O {np.corrcoef(g.t_off_rush, r)[0, 1]:+.3f}  (n={len(g)})")
    weeks = sorted(d.week.unique())
    for base, lab in (("blend|base", "our 50-50 blend"), ("sl", "Sleeper")):
        lowo = pd.concat([correction(d, base, d.week != w, d.week == w) for w in weeks])
        fwd = pd.concat([correction(d, base, d.week < w, d.week == w) for w in weeks])
        d[f"{base}+lowo"], d[f"{base}+fwd"] = d[base] + lowo.reindex(d.index), d[base] + fwd.reindex(d.index)
        x, lo, hi = vs(d, f"{base}+lowo", base)
        print(f"\n{lab} + FTN correction, leave-one-week-out: MAE change {x:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)}; negative = helps)")
        f = d.dropna(subset=[f"{base}+fwd"])
        x, lo, hi = vs(f, f"{base}+fwd", base)
        print(f"{lab} + FTN correction, forward only (weeks {f.week.min()}-{f.week.max()}): MAE change {x:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(f)})")
        for pos, g in d.groupby("position"):
            x, lo, hi = vs(g, f"{base}+lowo", base)
            print(f"   {pos} (leave-one-week-out): {x:+.3f} [{lo:+.3f}, {hi:+.3f}]")
