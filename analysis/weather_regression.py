"""Basic regression: does weather move team rushing / receiving points? (2025, game level)

Uses the roof / temp (F) / wind (mph) columns in the nflverse schedule. Rain and snow are not in that
file; they need an outside weather source (e.g. Open-Meteo) - see the README note.

Dependent variable per game, summed over both teams:
  * residual = actual points - the model's leave-one-week-out prediction  (controls for team strength)
  * raw      = actual points
OLS with normal-approx p-values; one row per game, so rows are independent.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from nflfp import backtest, data, model

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025


def ols(y: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    A = np.c_[np.ones(len(X)), X.to_numpy(float)]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = len(y) - A.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = [math.erfc(abs(v) / math.sqrt(2)) for v in t]  # two-sided, normal approximation
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    out = pd.DataFrame({"coef": beta, "se": se, "p": p}, index=["const", *X.columns])
    out.attrs["r2"], out.attrs["n"] = r2, len(y)
    return out


def main() -> None:
    pd.options.display.float_format = "{:.3f}".format
    s = model.Scoring()
    bt = backtest.walk_forward(data.load_player_weeks(SEASON), s, 1, leave_one_out=True)
    bt["res_rush"], bt["res_rec"] = bt.act_rush - bt.mod_rush, bt.act_rec - bt.mod_rec

    g = data.load_schedule(SEASON)
    g = g[g.result.notna()].copy()
    parts = []
    for r in g.itertuples():
        rows = bt[(bt.week == r.week) & (bt.team.isin([r.home_team, r.away_team]))]
        parts.append(dict(game_id=r.game_id, n=len(rows), **rows[["act_rush", "act_rec", "res_rush", "res_rec"]].sum().to_dict()))
    gm = g.merge(pd.DataFrame(parts), on="game_id")
    gm = gm[gm.n == 2]

    gm["indoor"] = gm.roof.isin(["dome", "closed"]).astype(float)
    out = gm[gm.indoor == 0].dropna(subset=["temp", "wind"])  # outdoor games with a reading
    ind = gm[(gm.indoor == 1) | gm.index.isin(out.index)].copy()
    ind["temp_c"] = (ind.temp - 60) / 10          # per 10F vs a mild 60F
    ind["wind_c"] = ind.wind / 5                  # per 5 mph
    ind["cold"] = (ind.temp < 40).astype(float)   # NaN temp (indoor) -> False
    ind["hot"] = (ind.temp >= 80).astype(float)
    ind["windy"] = (ind.wind >= 15).astype(float)
    for c in ("temp_c", "wind_c"):
        ind[c] = ind[c].fillna(0.0)               # indoors: weather has no effect

    print(f"{SEASON}: {len(ind)} games ({int(ind.indoor.sum())} indoor, {len(ind) - int(ind.indoor.sum())} outdoor); "
          f"outdoor: {int(ind.cold.sum())} cold (<40F), {int(ind.hot.sum())} hot (>=80F), {int(ind.windy.sum())} windy (>=15mph)\n")
    specs = {"continuous (per 10F above 60, per 5mph wind)": ["indoor", "temp_c", "wind_c"],
             "buckets (vs mild outdoor game)": ["indoor", "cold", "hot", "windy"]}
    for name, cols in specs.items():
        print(f"### {name}")
        for dv, label in (("res_rush", "rushing pts, both teams, vs model"), ("res_rec", "receiving pts, both teams, vs model"),
                          ("act_rush", "rushing pts, both teams, raw"), ("act_rec", "receiving pts, both teams, raw")):
            t = ols(ind[dv].to_numpy(float), ind[cols])
            print(f"-- {label}   (n={t.attrs['n']}, R2={t.attrs['r2']:.3f}, sd of dv={ind[dv].std():.1f})")
            print(t.drop(index="const").to_string())
        print()


if __name__ == "__main__":
    main()
