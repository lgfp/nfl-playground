"""Game-environment adjustment: betting lines (point spread, game total) and wind.

The matchup model knows nothing about how a game is expected to go. Pre-game lines carry some of
that (and some team quality / pace); wind suppresses passing. In backtests, lines lift team-level R2
from ~0.04 to ~0.06 for rushing and ~0.13 to ~0.16 for receiving.

For rushing and for receiving points we fit, on the season's past team-games,

    actual - model expectation = b0 + b_fav * fav + b_total * (total_line - mean total)      [rushing]
                                 + b_wind * (wind5 - mean wind5)                              [receiving only]

where `fav` is how many points the team is favoured by and `wind5` is the game-time wind in units of
5 mph (0 in a dome / closed roof). The result is added to the matchup projection. Quarterback passing
points follow the receiving adjustment (same passes).

If a line or a wind forecast is missing for a team, that team is left unadjusted for it (and recorded in
`LineAdjustment.missing` / `.missing_wind`) instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import model

FEATURES = ("fav", "total")   # betting-line features (both rushing and receiving)
WIND = "wind"                 # receiving only


def _indoor(schedule: pd.DataFrame) -> pd.Series:
    """Indoor games (dome / closed roof). A missing roof entry falls back to the home team's usual roof."""
    roof = schedule["roof"]
    usual = roof.groupby(schedule["home_team"]).transform(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    return roof.fillna(usual).isin(["dome", "closed"])


def team_lines(schedule: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (week, team): `fav` (points favoured by; negative = underdog), `total_line`, and
    `wind5` (wind in 5 mph units, 0 indoors, NaN when unknown) if `weather` (see weather.game_weather) is given."""
    base = pd.DataFrame({"week": schedule["week"], "total_line": schedule["total_line"]})
    home = base.assign(team=schedule["home_team"], fav=schedule["spread_line"])
    away = base.assign(team=schedule["away_team"], fav=-schedule["spread_line"])
    out = pd.concat([home, away], ignore_index=True)
    if weather is not None:
        wind = weather["wind_mph"].reindex(schedule["game_id"]).to_numpy()
        wind5 = np.where(_indoor(schedule).to_numpy(), 0.0, wind / 5.0)
        out["wind5"] = np.concatenate([wind5, wind5])
    return out


@dataclass
class LineAdjustment:
    coef: dict[str, np.ndarray]          # "rush" / "rec" -> [b0, b_feature...] in the order of `use[kind]`
    use: dict[str, tuple[str, ...]]
    total_mean: float
    wind_mean: float
    lines: pd.DataFrame                  # indexed (week, team)
    missing: set = field(default_factory=set)       # (week, team) with no betting line
    missing_wind: set = field(default_factory=set)  # (week, team) with no wind (treated as average wind)

    def _x(self, kind: str, fav: float, total: float, wind5: float) -> np.ndarray:
        vals = {"fav": fav, "total": total - self.total_mean, "wind": wind5 - self.wind_mean}
        return np.array([1.0] + [vals[f] for f in self.use[kind]])

    def adjust(self, week: int, offense: str, rush: float, rec: float) -> tuple[float, float]:
        """Add the adjustment to a team's rushing / receiving points (floored at 0)."""
        key = (week, offense)
        try:
            row = self.lines.loc[key]
        except KeyError:
            row = None
        if row is None or pd.isna(row["fav"]) or pd.isna(row["total_line"]):
            self.missing.add(key)  # lines are posted close to game day: leave this team unadjusted
            return rush, rec
        wind5 = row["wind5"] if "wind5" in row.index else np.nan
        if "wind" in self.use["rec"] and pd.isna(wind5):
            self.missing_wind.add(key)
            wind5 = self.wind_mean  # unknown conditions: average wind, i.e. no wind adjustment
        fav, total = float(row["fav"]), float(row["total_line"])
        d_rush = float(self.coef["rush"] @ self._x("rush", fav, total, float(wind5)))
        d_rec = float(self.coef["rec"] @ self._x("rec", fav, total, float(wind5)))
        return max(rush + d_rush, 0.0), max(rec + d_rec, 0.0)


def fit_line_adjustment(
    tg: pd.DataFrame, ratings: model.Ratings, scoring: model.Scoring, lines: pd.DataFrame,
    use: tuple[str, ...] = FEATURES, wind: bool = True,
) -> LineAdjustment:
    """Fit the adjustment on the team-games `tg` (as from model.team_games) that have lines.

    `use` picks the betting-line features; `wind` adds the wind term to receiving (needs a `wind5` column in `lines`).
    """
    use = tuple(use)
    if not use and not wind:
        raise ValueError("nothing to fit: pass line features and/or wind")
    bad = set(use) - set(FEATURES)
    if bad:
        raise ValueError(f"use must be a subset of {FEATURES}")
    has_wind = wind and "wind5" in lines.columns
    d = tg.merge(lines, left_on=["week", "off_team"], right_on=["week", "team"], how="inner").dropna(subset=["fav", "total_line"])
    total_mean = d["total_line"].mean()
    wind_mean = float(d["wind5"].mean()) if has_wind and d["wind5"].notna().any() else 0.0
    exp = pd.DataFrame({c: ratings.mu[c] * ratings.dfn.loc[d["def_team"], c].to_numpy() * ratings.off.loc[d["off_team"], c].to_numpy()
                        for c in model.COMPONENTS})
    exp_rush, exp_rec = model.points(exp, scoring)
    act_rush, act_rec = model.points(d.reset_index(drop=True), scoring)
    cols = {"fav": d["fav"].to_numpy(float), "total": (d["total_line"] - total_mean).to_numpy(float)}
    if has_wind:
        cols["wind"] = (d["wind5"] - wind_mean).to_numpy(float)
    feats = {"rush": use, "rec": use + ((WIND,) if has_wind else ())}
    coef = {}
    for kind, act, e in (("rush", act_rush, exp_rush), ("rec", act_rec, exp_rec)):
        y = act.to_numpy() - e.to_numpy()
        X = np.c_[np.ones(len(d)), *[cols[f] for f in feats[kind]]]
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)  # drop games without wind data for this fit only
        coef[kind] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return LineAdjustment(coef=coef, use=feats, total_mean=total_mean, wind_mean=wind_mean, lines=lines.set_index(["week", "team"]))
