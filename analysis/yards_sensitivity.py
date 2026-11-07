"""Sensitivity of module 1: Vegas modes, and ridge strength x last-season weight x prior basis (walk-forward 2025-2026).

    uv run python analysis/yards_sensitivity.py
"""
import itertools

import numpy as np
import pandas as pd

from analysis.yards_backtest import MAIN_FROM, load, metrics, walk_forward
from nflfp import yards as Y

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 220
games, fixtures = load()

print("=== A. Vegas modes, weeks 4+ (R2 / MAE; 'none' = the model without the lines adjustment)")
rows = []
for mode in ("transfer", "free", "relative", "unbalanced", "unbalanced_flat"):
    bt = walk_forward(games, fixtures, Y.YardsConfig(lines_mode=mode))
    sub = bt[bt.week >= MAIN_FROM]
    row = {"mode": mode}
    for side in Y.SIDES:
        for v in ("ridge", "ridge_vegas", "plain", "plain_vegas"):
            m = metrics(sub, side, f"{side}_{v}"); row[f"{side} {v} R2"] = m["R2"]; row[f"{side} {v} MAE"] = m["MAE"]
    sl = sub.groupby(["season", "week"])[["lines_rush_ridge", "lines_pass_ridge"]].first().dropna().tail(1).iloc[0]
    row["latest slopes (rush, pass)"] = f"{sl.lines_rush_ridge:+.3f}, {sl.lines_pass_ridge:+.3f}"
    rows.append(row)
print(pd.DataFrame(rows).set_index("mode").T.to_string())

print("\n=== B. Ridge grid (no Vegas): R2 weeks 4+ for rush / pass (and weeks 1-3)")
rows = []
for basis, w, k in itertools.product(("adjusted", "raw"), (0.0, 0.15, 0.3, 0.5, 0.75), (2, 4, 8, 16, 32)):
    bt = walk_forward(games, fixtures, Y.YardsConfig(ridge_strength=k, prior_weight=w, prior_basis=basis), only=("ridge",))
    a, b = bt[bt.week >= MAIN_FROM], bt[bt.week < MAIN_FROM]
    r = {"basis": basis, "last-season weight": w, "strength": k}
    for side in Y.SIDES:
        r[f"{side} R2 (4+)"] = metrics(a, side, f"{side}_ridge")["R2"]; r[f"{side} R2 (1-3)"] = metrics(b, side, f"{side}_ridge")["R2"]
    r["sum R2 (4+)"] = r["rush R2 (4+)"] + r["pass R2 (4+)"]
    rows.append(r)
g = pd.DataFrame(rows)
print(g.sort_values("sum R2 (4+)", ascending=False).head(12).to_string(index=False))
print("\nbest strength for each last-season weight (adjusted basis), sum of R2 weeks 4+:")
adj = g[g.basis == "adjusted"].pivot(index="last-season weight", columns="strength", values="sum R2 (4+)"); print(adj.to_string())
print("\nsame for the raw basis:"); print(g[g.basis == "raw"].pivot(index="last-season weight", columns="strength", values="sum R2 (4+)").to_string())
