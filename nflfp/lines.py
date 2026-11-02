"""Optional betting-line adjustment (point spread and game total).

The matchup model knows nothing about how a game is expected to go. Pre-game lines carry some of
that (and some team quality / pace). In backtests, adding them lifted team-level R2 from ~0.04 to
~0.06 for rushing and ~0.13 to ~0.16 for receiving - small, so it is OFF by default.

For each of rushing and receiving points we fit, on past team-games,

    actual - model expectation = b0 + b_fav * fav + b_total * (total_line - mean total)

where `fav` is how many points the team is favoured by, and add that to the matchup projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model

FEATURES = ("fav", "total")


def team_lines(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (week, team): `fav` (points favoured by; negative = underdog) and `total_line`."""
    home = pd.DataFrame({"week": schedule["week"], "team": schedule["home_team"],
                         "fav": schedule["spread_line"], "total_line": schedule["total_line"]})
    away = pd.DataFrame({"week": schedule["week"], "team": schedule["away_team"],
                         "fav": -schedule["spread_line"], "total_line": schedule["total_line"]})
    return pd.concat([home, away], ignore_index=True)


@dataclass
class LineAdjustment:
    coef: dict[str, np.ndarray]   # "rush" / "rec" -> [b0, b_feature...]
    use: tuple[str, ...]
    total_mean: float
    lines: pd.DataFrame           # (week, team) -> fav, total_line, indexed for lookup

    def _x(self, fav: float, total: float) -> np.ndarray:
        vals = {"fav": fav, "total": total - self.total_mean}
        return np.array([1.0] + [vals[f] for f in self.use])

    def delta(self, kind: str, fav: float, total: float) -> float:
        return float(self.coef[kind] @ self._x(fav, total))

    def adjust(self, week: int, offense: str, rush: float, rec: float) -> tuple[float, float]:
        """Add the line adjustment to a team's rushing / receiving points (floored at 0)."""
        try:
            row = self.lines.loc[(week, offense)]
        except KeyError:
            raise SystemExit(f"No betting line for {offense} in week {week} (lines are posted close to game day).")
        if row.isna().any():
            raise SystemExit(f"No betting line for {offense} in week {week} (lines are posted close to game day).")
        fav, total = float(row["fav"]), float(row["total_line"])
        return max(rush + self.delta("rush", fav, total), 0.0), max(rec + self.delta("rec", fav, total), 0.0)


def fit_line_adjustment(
    tg: pd.DataFrame, ratings: model.Ratings, scoring: model.Scoring, lines: pd.DataFrame,
    use: tuple[str, ...] = FEATURES,
) -> LineAdjustment:
    """Fit the adjustment on the team-games `tg` (as from model.team_games) that have lines."""
    use = tuple(use)
    bad = set(use) - set(FEATURES)
    if bad or not use:
        raise ValueError(f"use must be a non-empty subset of {FEATURES}")
    d = tg.merge(lines, left_on=["week", "off_team"], right_on=["week", "team"], how="inner").dropna(subset=["fav", "total_line"])
    total_mean = d["total_line"].mean()
    exp = pd.DataFrame({c: ratings.mu[c] * ratings.dfn.loc[d["def_team"], c].to_numpy() * ratings.off.loc[d["off_team"], c].to_numpy()
                        for c in model.COMPONENTS})
    exp_rush, exp_rec = model.points(exp, scoring)
    act_rush, act_rec = model.points(d.reset_index(drop=True), scoring)
    X = np.c_[np.ones(len(d)), *[d["fav"] if f == "fav" else d["total_line"] - total_mean for f in use]]
    coef = {k: np.linalg.lstsq(X, (act.to_numpy() - e.to_numpy()), rcond=None)[0]
            for k, act, e in (("rush", act_rush, exp_rush), ("rec", act_rec, exp_rec))}
    return LineAdjustment(coef=coef, use=use, total_mean=total_mean, lines=lines.set_index(["week", "team"]))
