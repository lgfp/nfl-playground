"""Basic regression: does weather move team rushing / receiving points? (2025, game level)

Weather is the 3 hours after kickoff at the home stadium, from Open-Meteo (nflfp/weather.py); roof comes
from the nflverse schedule (dome / closed = indoor). Neutral-site games are excluded.

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

from nflfp import backtest, data, model, weather

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
DROP = sys.argv[2:]  # optional game_ids to leave out, as a robustness check


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

    gm = gm.merge(weather.check_complete(weather.game_weather(g)).reset_index(), on="game_id")  # drops neutral-site games
    gm = gm[~gm.game_id.isin(DROP)]
    gm["indoor"] = gm.roof.isin(["dome", "closed"]).astype(float)
    ind = gm.copy()
    out = ind.indoor == 0
    # weather only matters outdoors: zero everything for indoor games
    ind["temp_c"] = np.where(out, (ind.temp_f - 60) / 10, 0.0)   # per 10F vs a mild 60F
    ind["wind_c"] = np.where(out, ind.wind_mph / 5, 0.0)          # per 5 mph
    ind["rain_mm"] = np.where(out, ind.precip_mm, 0.0)            # mm in the 3h after kickoff
    ind["cold"] = (out & (ind.temp_f < 40)).astype(float)
    ind["hot"] = (out & (ind.temp_f >= 80)).astype(float)
    ind["windy"] = (out & (ind.wind_mph >= 15)).astype(float)
    ind["wet"] = (out & (ind.precip_mm >= 1.0)).astype(float)     # >= 1 mm in 3h
    ind["snow"] = (out & (ind.snow_cm > 0)).astype(float)

    print(f"{SEASON}: {len(ind)} games ({int(ind.indoor.sum())} indoor, {int(out.sum())} outdoor); outdoor: "
          f"{int(ind.cold.sum())} cold (<40F), {int(ind.hot.sum())} hot (>=80F), {int(ind.windy.sum())} windy (>=15mph), "
          f"{int(ind.wet.sum())} wet (>=1mm), {int(ind.snow.sum())} with snow\n")
    specs = {"continuous (per 10F above 60, per 5mph wind, per mm rain)": ["indoor", "temp_c", "wind_c", "rain_mm", "snow"],
             "buckets (vs a dry mild outdoor game)": ["indoor", "cold", "hot", "windy", "wet", "snow"]}
    for name, cols in specs.items():
        print(f"### {name}")
        for dv, label in (("res_rush", "rushing pts, both teams, vs model"), ("res_rec", "receiving pts, both teams, vs model"),
                          ("act_rush", "rushing pts, both teams, raw"), ("act_rec", "receiving pts, both teams, raw")):
            use = [c for c in cols if ind[c].std() > 0]  # e.g. no snow games left after --drop
            t = ols(ind[dv].to_numpy(float), ind[use])
            print(f"-- {label}   (n={t.attrs['n']}, R2={t.attrs['r2']:.3f}, sd of dv={ind[dv].std():.1f})")
            print(t.drop(index="const").to_string())
        print()


if __name__ == "__main__":
    main()
