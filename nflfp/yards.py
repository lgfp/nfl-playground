"""Module 1 - expected team rushing and passing yards for a week.

For every game we produce eight numbers per team: {rush, pass} x {plain, ridge} x {no Vegas, Vegas}.

The core is an iterative, opponent-adjusted, multiplicative model, fitted separately for rushing and passing yards:

    expected yards (offense o vs defense d) = MU * OFF[o] * DEF[d]

MU is the league mean per team-game, OFF / DEF are factors around 1.0 (OFF > 1: gains more than an average offense
would against an average defense; DEF > 1: allows more than average). They are found by alternating updates -
each defense's factor is the mean of its games' yards after removing the strength of the offense it faced, then
each offense's factor the same way - until they stop changing. Both sets are normalised to mean 1.

* plain: only the current season's earlier games. A team with no games yet gets 1.0.
* ridge: each factor is pulled toward a *target* by `ridge_strength` pseudo-games of the target:

      factor = (sum of the team's opponent-adjusted ratios + strength * target) / (games + strength)
      target = prior_weight * (team's previous-season factor) + (1 - prior_weight) * 1.0

  i.e. by default 30% last season's team and 70% the current league average. At week 1 (no games yet) the ridge
  model is simply MU(last season) * target factors.
* Vegas: favourites are expected to run more and pass less. Three modes, all fitted by least squares through the origin
  on the residuals of the previous season (in-sample) and the current season's earlier games:
    "transfer" (the spec): add `beta * fav` rushing yards and remove `beta * fav` passing yards;
    "free":     separate yard slopes for rushing and passing (do favourites really pass less?);
    "relative": rush x (1 + a * fav) and pass x (1 + b * fav) - separate relative slopes.

Everything uses only games before the week being projected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIDES = ("rush", "pass")
YARD_COLS = {"rush": "rushing_yards", "pass": "passing_yards"}


@dataclass(frozen=True)
class YardsConfig:
    ridge_strength: float = 8.0      # pseudo-games of the target mixed into every team's factor
    prior_weight: float = 0.3        # weight of last season's team factor in the ridge target (rest: league average)
    prior_basis: str = "raw"         # last season's factor: "raw" (team average / league average) or "adjusted" (opponent-adjusted fit)
    lines_mode: str = "transfer"     # "transfer": +beta*fav rush / -beta*fav pass; "free": separate slopes
    max_iter: int = 2000
    tol: float = 1e-10


# --------------------------------------------------------------------------------------------------- data helpers
def team_game_yards(player_week: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense, defense): rush_yds and pass_yds summed over the offense's players."""
    keys = ["season", "week", "team", "opponent_team"] if "season" in player_week else ["week", "team", "opponent_team"]
    g = player_week.groupby(keys, as_index=False)[list(YARD_COLS.values())].sum()
    g = g.rename(columns={"team": "off_team", "opponent_team": "def_team", "rushing_yards": "rush_yds", "passing_yards": "pass_yds"})
    if "season" not in g:
        g.insert(0, "season", 0)
    return g


def fixtures_from_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense, defense) with `fav` = points the offense is favoured by (negative: underdog).

    nflverse `spread_line` is positive when the home team is favoured.
    """
    base = schedule[["season", "week"]]
    home = base.assign(off_team=schedule["home_team"], def_team=schedule["away_team"], fav=schedule["spread_line"])
    away = base.assign(off_team=schedule["away_team"], def_team=schedule["home_team"], fav=-schedule["spread_line"])
    return pd.concat([home, away], ignore_index=True)


# ------------------------------------------------------------------------------------------------------ the fit
@dataclass
class Ratings:
    mu: dict[str, float]
    off: pd.DataFrame        # index team, columns rush / pass
    dfn: pd.DataFrame
    n_games: int
    iterations: dict[str, int]

    def expected(self, off_team: str, def_team: str) -> dict[str, float]:
        return {s: self.mu[s] * self.off.loc[off_team, s] * self.dfn.loc[def_team, s] for s in SIDES}


@dataclass
class Prior:
    """Last season's league means and team factors, and the ridge targets built from them."""
    mu: dict[str, float]
    off_factor: pd.DataFrame
    dfn_factor: pd.DataFrame

    def targets(self, weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return weight * self.off_factor + (1 - weight), weight * self.dfn_factor + (1 - weight)


def _fit_side(o, d, y, n, mu, strength, t_off, t_dfn, max_iter, tol) -> tuple[np.ndarray, np.ndarray, int]:
    off, dfn = t_off.copy(), t_dfn.copy()
    n_o = np.bincount(o, minlength=n).astype(float)
    n_d = np.bincount(d, minlength=n).astype(float)
    it = 0
    for it in range(1, max_iter + 1):
        num = np.bincount(d, y / (mu * off[o]), n) + strength * t_dfn
        den = n_d + strength
        new_dfn = np.divide(num, den, out=t_dfn.copy(), where=den > 0)
        new_dfn = new_dfn / new_dfn.mean()
        num = np.bincount(o, y / (mu * new_dfn[d]), n) + strength * t_off
        den = n_o + strength
        new_off = np.divide(num, den, out=t_off.copy(), where=den > 0)
        new_off = new_off / new_off.mean()
        delta = max(np.abs(new_dfn - dfn).max(), np.abs(new_off - off).max())
        off, dfn = new_off, new_dfn
        if delta < tol:
            break
    return off, dfn, it


def fit_ratings(games: pd.DataFrame, teams: list[str], cfg: YardsConfig, prior: Prior | None = None,
                ridge: bool = False, fallback_mu: dict[str, float] | None = None) -> Ratings:
    """Fit MU / OFF / DEF on `games` (columns off_team, def_team, rush_yds, pass_yds).

    ridge=True pulls every team toward the targets built from `prior` (needs one); ridge=False is the plain fit.
    With no games, MU comes from `fallback_mu` (last season's league mean) and the factors are the targets.
    """
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if ridge and prior is not None:
        t_off, t_dfn = (x.reindex(teams).fillna(1.0) for x in prior.targets(cfg.prior_weight))
        strength = cfg.ridge_strength
    else:
        t_off = t_dfn = pd.DataFrame(1.0, index=teams, columns=SIDES)
        strength = 0.0
    off_out, dfn_out, mu, iters = {}, {}, {}, {}
    o = games["off_team"].map(ix).to_numpy()
    d = games["def_team"].map(ix).to_numpy()
    for s in SIDES:
        y = games[f"{s}_yds"].to_numpy(float)
        if len(y) == 0:
            if fallback_mu is None:
                raise ValueError("no games to fit and no fallback league mean")
            mu[s] = fallback_mu[s]
            off_out[s], dfn_out[s], iters[s] = t_off[s].to_numpy(), t_dfn[s].to_numpy(), 0
            continue
        mu[s] = float(y.mean())
        off_out[s], dfn_out[s], iters[s] = _fit_side(o, d, y, n, mu[s], strength, t_off[s].to_numpy(), t_dfn[s].to_numpy(), cfg.max_iter, cfg.tol)
    return Ratings(mu, pd.DataFrame(off_out, index=teams), pd.DataFrame(dfn_out, index=teams), len(games), iters)


def build_prior(prev_games: pd.DataFrame, teams: list[str], cfg: YardsConfig) -> Prior:
    """League means and team factors from a finished season ("adjusted": a plain opponent-adjusted fit; "raw": plain averages)."""
    if cfg.prior_basis == "adjusted":
        r = fit_ratings(prev_games, teams, cfg)
        return Prior(r.mu, r.off, r.dfn)
    if cfg.prior_basis != "raw":
        raise ValueError("prior_basis must be 'adjusted' or 'raw'")
    mu = {s: float(prev_games[f"{s}_yds"].mean()) for s in SIDES}
    off = pd.DataFrame({s: prev_games.groupby("off_team")[f"{s}_yds"].mean() / mu[s] for s in SIDES}).reindex(teams).fillna(1.0)
    dfn = pd.DataFrame({s: prev_games.groupby("def_team")[f"{s}_yds"].mean() / mu[s] for s in SIDES}).reindex(teams).fillna(1.0)
    return Prior(mu, off, dfn)


# ---------------------------------------------------------------------------------------------- Vegas adjustment
@dataclass
class LinesAdjustment:
    mode: str
    beta: float = 0.0                                   # transfer: yards moved from passing to rushing per point favoured
    slopes: dict[str, float] = field(default_factory=dict)   # free: yards per point favoured; relative: fraction per point

    def as_slopes(self) -> tuple[float, float]:
        """(rush, pass) slope per point favoured: yards for transfer / free, a fraction for relative."""
        return (self.beta, -self.beta) if self.mode == "transfer" else (self.slopes["rush"], self.slopes["pass"])

    def apply(self, rush: float, pas: float, fav: float) -> tuple[float, float]:
        if self.mode == "transfer":
            return max(rush + self.beta * fav, 0.0), max(pas - self.beta * fav, 0.0)
        if self.mode == "free":
            return max(rush + self.slopes["rush"] * fav, 0.0), max(pas + self.slopes["pass"] * fav, 0.0)
        return max(rush * (1 + self.slopes["rush"] * fav), 0.0), max(pas * (1 + self.slopes["pass"] * fav), 0.0)


def fit_lines(resid: pd.DataFrame, mode: str) -> LinesAdjustment:
    """`resid`: fav, rush_resid, pass_resid (actual - expected yards) and, for "relative", rush_exp / pass_exp.
    No intercept: zero spread, zero adjustment."""
    f, rr, pr = resid["fav"].to_numpy(float), resid["rush_resid"].to_numpy(float), resid["pass_resid"].to_numpy(float)
    ss = float((f ** 2).sum())
    if ss == 0:
        return LinesAdjustment(mode, 0.0, {"rush": 0.0, "pass": 0.0})
    if mode == "relative":      # resid = exp * slope * f  ->  slope = sum(f exp resid) / sum((f exp)^2)
        er, ep = resid["rush_exp"].to_numpy(float), resid["pass_exp"].to_numpy(float)
        return LinesAdjustment(mode, slopes={"rush": float((f * er * rr).sum() / ((f * er) ** 2).sum()),
                                             "pass": float((f * ep * pr).sum() / ((f * ep) ** 2).sum())})
    if mode == "transfer":      # minimise sum (rr - b f)^2 + (pr + b f)^2
        return LinesAdjustment(mode, beta=float((f * (rr - pr)).sum() / (2 * ss)))
    if mode == "free":
        return LinesAdjustment(mode, slopes={"rush": float((f * rr).sum() / ss), "pass": float((f * pr).sum() / ss)})
    raise ValueError("lines_mode must be 'transfer', 'free' or 'relative'")


# ------------------------------------------------------------------------------------------------ the projection
@dataclass
class SeasonContext:
    """Things that depend only on the season, computed once: last season's prior and its in-sample residuals."""
    teams: list[str]
    prior: Prior | None
    prev_resid: pd.DataFrame | None


def _residuals(games: pd.DataFrame, fixtures: pd.DataFrame, r: Ratings) -> pd.DataFrame:
    m = games.merge(fixtures[["season", "week", "off_team", "def_team", "fav"]], on=["season", "week", "off_team", "def_team"], how="inner")
    exp = pd.DataFrame([r.expected(a, b) for a, b in zip(m.off_team, m.def_team)], index=m.index)
    return pd.DataFrame({"fav": m.fav, "rush_resid": m.rush_yds - exp["rush"], "pass_resid": m.pass_yds - exp["pass"],
                         "rush_exp": exp["rush"], "pass_exp": exp["pass"]}).dropna()


def season_context(games: pd.DataFrame, fixtures: pd.DataFrame, season: int, cfg: YardsConfig) -> SeasonContext:
    teams = sorted(set(games.loc[games.season == season, "off_team"]) | set(fixtures.loc[fixtures.season == season, "off_team"]) |
                   set(games.loc[games.season == season - 1, "off_team"]))
    prev = games[games.season == season - 1]
    if prev.empty:
        return SeasonContext(teams, None, None)
    prior = build_prior(prev, teams, cfg)
    plain_prev = fit_ratings(prev, teams, cfg)   # in-sample residuals of last season, used to fit the Vegas slope
    return SeasonContext(teams, prior, _residuals(prev, fixtures, plain_prev))


VARIANTS = [("plain", False, False), ("ridge", True, False), ("plain_vegas", False, True), ("ridge_vegas", True, True)]
OUT_COLS = [f"{s}_{v}" for s in SIDES for v, _, _ in VARIANTS]   # the eight numbers


def project_week(games: pd.DataFrame, fixtures: pd.DataFrame, season: int, week: int, cfg: YardsConfig = YardsConfig(),
                 ctx: SeasonContext | None = None, only: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The eight yard projections for every game of `week`, using only earlier games.

    games: team_game_yards() output for this and the previous season; fixtures: fixtures_from_schedule() output.
    Returns one row per (offense, defense) with fav and the columns rush_plain, rush_ridge, rush_plain_vegas,
    rush_ridge_vegas, pass_plain, pass_ridge, pass_plain_vegas, pass_ridge_vegas. `only` restricts the variants
    computed (e.g. ("ridge",)) when the others are not needed.
    """
    ctx = ctx or season_context(games, fixtures, season, cfg)
    cur = games[(games.season == season) & (games.week < week)]
    fx = fixtures[(fixtures.season == season) & (fixtures.week == week)].reset_index(drop=True)
    fallback = ctx.prior.mu if ctx.prior is not None else None
    fits = {
        "plain": fit_ratings(cur, ctx.teams, cfg, fallback_mu=fallback),
        "ridge": fit_ratings(cur, ctx.teams, cfg, ctx.prior, ridge=True, fallback_mu=fallback) if ctx.prior is not None else None,
    }
    exp = {k: pd.DataFrame([r.expected(a, b) for a, b in zip(fx.off_team, fx.def_team)]) if r is not None else None for k, r in fits.items()}
    out = fx[["season", "week", "off_team", "def_team", "fav"]].copy()
    out.attrs["lines"] = {}    # the fitted Vegas adjustment of each model, for inspection
    for variant, ridge, vegas in VARIANTS:
        if only is not None and variant not in only:
            continue
        base = "ridge" if ridge else "plain"
        e = exp[base]
        if e is None:                       # no previous season: a ridge model cannot be built
            for s in SIDES:
                out[f"{s}_{variant}"] = np.nan
            continue
        rush, pas = e["rush"].to_numpy(), e["pass"].to_numpy()
        if vegas:
            parts = [ctx.prev_resid] if ctx.prev_resid is not None else []
            if len(cur):
                parts.append(_residuals(cur, fixtures, fits[base]))
            empty = pd.DataFrame({"fav": [], "rush_resid": [], "pass_resid": [], "rush_exp": [], "pass_exp": []})
            adj = fit_lines(pd.concat(parts, ignore_index=True) if parts else empty, cfg.lines_mode)
            out.attrs["lines"][base] = adj
            pairs = [adj.apply(a, b, f) for a, b, f in zip(rush, pas, out.fav.fillna(0.0))]
            rush, pas = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        out[f"rush_{variant}"], out[f"pass_{variant}"] = rush, pas
    return out
