"""Module 1b - expected team rushing and passing yards, built from plays.

Same eight outputs as module 1 ({rush, pass} x {plain, ridge} x {no Vegas, Vegas}), but a team's yards in a game are
split into the two things they are made of:

    yards = plays (volume) x yards per play (efficiency)

and each is opponent-adjusted separately, with the same iterative multiplicative engine as module 1:

    rush_att, pass_att   how many runs / pass plays (sacks included) a team calls    -> MU * OFF * DEF   (a team's style and game script)
    rush_ypc, pass_ypa   gross yards per run / per pass play                          -> MU * OFF * DEF   (where the opponent matters most)

Efficiency is averaged over plays (a game with 30 carries counts more than one with 8). Ridge pulls every factor toward a
target of `prior_weight` last-season team factor + (1 - prior_weight) league average, with separate strengths for
volume (stable, so it can lean on the team's own games) and efficiency (noisy).

The Vegas adjustment now acts on *volume*, where a spread should show up: a favourite runs more and passes less.
    "transfer": +gamma*fav run plays and -gamma*fav pass plays;
    "relative": run plays x (1 + a*fav), pass plays x (1 + b*fav).
Yards = adjusted plays x the opponent-adjusted yards per play.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import yards as Y

SIDES = Y.SIDES
QUANTS = ("rush_att", "pass_att", "rush_ypc", "pass_ypa")
VOLUME, EFFICIENCY = ("rush_att", "pass_att"), ("rush_ypc", "pass_ypa")
SIDE_OF = {"rush_att": "rush", "rush_ypc": "rush", "pass_att": "pass", "pass_ypa": "pass"}


@dataclass(frozen=True)
class PlaysConfig:
    strength_volume: float = 16.0      # pseudo-games of the target mixed into plays-per-game factors (grid best: volume regresses more)
    strength_efficiency: float = 8.0   # same, for yards-per-play factors
    prior_weight: float = 0.3          # weight of last season's team factor in the ridge target (rest: league average)
    lines_mode: str = "transfer"       # Vegas on volume: "transfer" or "relative"
    max_iter: int = 2000
    tol: float = 1e-10


def team_game_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense, defense) from run/pass plays (data.load_pbp): attempts and gross yards.

    A run is a designed run or scramble (kneels and spikes are not in the input); a pass play counts sacks as plays with 0 yards.
    """
    r, p = pbp[pbp.play_type == "run"], pbp[pbp.play_type == "pass"]
    k = ["season", "week", "posteam", "defteam"]
    g = pd.DataFrame({"rush_att": r.groupby(k).size(), "rush_yds": r.groupby(k).rushing_yards.sum(),
                      "pass_att": p.groupby(k).size(), "pass_yds": p.groupby(k).passing_yards.sum()}).fillna(0.0).reset_index()
    return g.rename(columns={"posteam": "off_team", "defteam": "def_team"})


# ------------------------------------------------------------------------------------------------------ the fit
@dataclass
class Ratings:
    mu: dict[str, float]
    off: pd.DataFrame      # index team, columns QUANTS
    dfn: pd.DataFrame
    n_games: int

    def expected(self, off_team: str, def_team: str) -> dict[str, float]:
        e = {q: self.mu[q] * self.off.loc[off_team, q] * self.dfn.loc[def_team, q] for q in QUANTS}
        e["rush_yds"], e["pass_yds"] = e["rush_att"] * e["rush_ypc"], e["pass_att"] * e["pass_ypa"]
        return e


@dataclass
class Prior:
    mu: dict[str, float]
    off_factor: pd.DataFrame
    dfn_factor: pd.DataFrame

    def targets(self, weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return weight * self.off_factor + (1 - weight), weight * self.dfn_factor + (1 - weight)


def _series(games: pd.DataFrame, q: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(row mask, values, weights, league mean) for one quantity; efficiency is yards / plays, weighted by plays."""
    if q in VOLUME:
        y = games[q].to_numpy(float)
        return np.ones(len(y), bool), y, np.ones(len(y)), float(y.mean()) if len(y) else np.nan
    side = SIDE_OF[q]
    att, yds = games[f"{side}_att"].to_numpy(float), games[f"{side}_yds"].to_numpy(float)
    ok = att > 0
    if not ok.any():
        return ok, np.array([]), np.array([]), np.nan
    return ok, yds[ok] / att[ok], att[ok] / att[ok].mean(), float(yds[ok].sum() / att[ok].sum())


def fit_ratings(games: pd.DataFrame, teams: list[str], cfg: PlaysConfig, prior: Prior | None = None, ridge: bool = False,
                fallback_mu: dict[str, float] | None = None) -> Ratings:
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if ridge and prior is not None:
        t_off, t_dfn = (x.reindex(teams).fillna(1.0) for x in prior.targets(cfg.prior_weight))
    else:
        t_off = t_dfn = pd.DataFrame(1.0, index=teams, columns=QUANTS)
    off_out, dfn_out, mu = {}, {}, {}
    o_all, d_all = games["off_team"].map(ix).to_numpy(), games["def_team"].map(ix).to_numpy()
    for q in QUANTS:
        ok, y, w, m = _series(games, q)
        if len(y) == 0:
            if fallback_mu is None:
                raise ValueError("no games to fit and no fallback league means")
            mu[q], off_out[q], dfn_out[q] = fallback_mu[q], t_off[q].to_numpy(), t_dfn[q].to_numpy()
            continue
        strength = (cfg.strength_volume if q in VOLUME else cfg.strength_efficiency) if (ridge and prior is not None) else 0.0
        mu[q] = m
        off_out[q], dfn_out[q], _ = Y._fit_side(o_all[ok], d_all[ok], y, n, m, strength, t_off[q].to_numpy(), t_dfn[q].to_numpy(), cfg.max_iter, cfg.tol, w)
    return Ratings(mu, pd.DataFrame(off_out, index=teams), pd.DataFrame(dfn_out, index=teams), len(games))


def build_prior(prev_games: pd.DataFrame, teams: list[str]) -> Prior:
    """League means and team factors (team average / league average) from a finished season."""
    mu, off, dfn = {}, {}, {}
    for q in QUANTS:
        if q in VOLUME:
            mu[q] = float(prev_games[q].mean())
            off[q] = prev_games.groupby("off_team")[q].mean() / mu[q]
            dfn[q] = prev_games.groupby("def_team")[q].mean() / mu[q]
        else:
            side = SIDE_OF[q]
            mu[q] = float(prev_games[f"{side}_yds"].sum() / prev_games[f"{side}_att"].sum())
            off[q] = prev_games.groupby("off_team")[f"{side}_yds"].sum() / prev_games.groupby("off_team")[f"{side}_att"].sum() / mu[q]
            dfn[q] = prev_games.groupby("def_team")[f"{side}_yds"].sum() / prev_games.groupby("def_team")[f"{side}_att"].sum() / mu[q]
    return Prior(mu, pd.DataFrame(off).reindex(teams).fillna(1.0), pd.DataFrame(dfn).reindex(teams).fillna(1.0))


# ------------------------------------------------------------------------------------- Vegas adjustment on volume
@dataclass
class VolumeLines:
    mode: str
    gamma: float = 0.0                                          # transfer: run plays gained (pass plays lost) per point favoured
    slopes: dict[str, float] = field(default_factory=dict)      # relative: fraction per point favoured, per side

    def as_slopes(self) -> tuple[float, float]:
        return (self.gamma, -self.gamma) if self.mode == "transfer" else (self.slopes["rush"], self.slopes["pass"])

    def apply(self, rush_att: float, pass_att: float, fav: float) -> tuple[float, float]:
        if self.mode == "transfer":
            return max(rush_att + self.gamma * fav, 0.0), max(pass_att - self.gamma * fav, 0.0)
        return max(rush_att * (1 + self.slopes["rush"] * fav), 0.0), max(pass_att * (1 + self.slopes["pass"] * fav), 0.0)


def fit_volume_lines(resid: pd.DataFrame, mode: str) -> VolumeLines:
    """`resid`: fav, rush_resid, pass_resid (actual - expected plays), rush_exp, pass_exp. Through the origin."""
    f = resid["fav"].to_numpy(float)
    rr, pr = resid["rush_resid"].to_numpy(float), resid["pass_resid"].to_numpy(float)
    ss = float((f ** 2).sum())
    if ss == 0:
        return VolumeLines(mode, 0.0, {"rush": 0.0, "pass": 0.0})
    if mode == "transfer":
        return VolumeLines(mode, gamma=float((f * (rr - pr)).sum() / (2 * ss)))
    if mode == "relative":
        er, ep = resid["rush_exp"].to_numpy(float), resid["pass_exp"].to_numpy(float)
        return VolumeLines(mode, slopes={"rush": float((f * er * rr).sum() / ((f * er) ** 2).sum()), "pass": float((f * ep * pr).sum() / ((f * ep) ** 2).sum())})
    raise ValueError("lines_mode must be 'transfer' or 'relative'")


# ------------------------------------------------------------------------------------------------ the projection
@dataclass
class SeasonContext:
    teams: list[str]
    prior: Prior | None
    prev_resid: pd.DataFrame | None


def _volume_residuals(games: pd.DataFrame, fixtures: pd.DataFrame, r: Ratings) -> pd.DataFrame:
    m = games.merge(fixtures[["season", "week", "off_team", "def_team", "fav"]], on=["season", "week", "off_team", "def_team"], how="inner")
    exp = pd.DataFrame([r.expected(a, b) for a, b in zip(m.off_team, m.def_team)], index=m.index)
    return pd.DataFrame({"fav": m.fav, "rush_resid": m.rush_att - exp["rush_att"], "pass_resid": m.pass_att - exp["pass_att"],
                         "rush_exp": exp["rush_att"], "pass_exp": exp["pass_att"]}).dropna()


def season_context(games: pd.DataFrame, fixtures: pd.DataFrame, season: int, cfg: PlaysConfig) -> SeasonContext:
    teams = sorted(set(games.loc[games.season == season, "off_team"]) | set(fixtures.loc[fixtures.season == season, "off_team"]) |
                   set(games.loc[games.season == season - 1, "off_team"]))
    prev = games[games.season == season - 1]
    if prev.empty:
        return SeasonContext(teams, None, None)
    return SeasonContext(teams, build_prior(prev, teams), _volume_residuals(prev, fixtures, fit_ratings(prev, teams, cfg)))


def project_week(games: pd.DataFrame, fixtures: pd.DataFrame, season: int, week: int, cfg: PlaysConfig = PlaysConfig(),
                 ctx: SeasonContext | None = None, only: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The eight yard projections (same column names as module 1) for every game of `week`, from earlier games only.

    games: team_game_plays() output for this and the previous season. The ridge model's expected plays and yards per play
    are also returned (rush_att_ridge, rush_ypc_ridge, ...).
    """
    ctx = ctx or season_context(games, fixtures, season, cfg)
    cur = games[(games.season == season) & (games.week < week)]
    fx = fixtures[(fixtures.season == season) & (fixtures.week == week)].reset_index(drop=True)
    fallback = ctx.prior.mu if ctx.prior is not None else None
    fits = {"plain": fit_ratings(cur, ctx.teams, cfg, fallback_mu=fallback),
            "ridge": fit_ratings(cur, ctx.teams, cfg, ctx.prior, ridge=True, fallback_mu=fallback) if ctx.prior is not None else None}
    exp = {k: pd.DataFrame([r.expected(a, b) for a, b in zip(fx.off_team, fx.def_team)]) if r is not None else None for k, r in fits.items()}
    out = fx[["season", "week", "off_team", "def_team", "fav"]].copy()
    out.attrs["lines"] = {}
    if exp["ridge"] is not None:
        for q in QUANTS:
            out[f"{q}_ridge"] = exp["ridge"][q].to_numpy()
    for variant, ridge, vegas in Y.VARIANTS:
        if only is not None and variant not in only:
            continue
        base = "ridge" if ridge else "plain"
        e = exp[base]
        if e is None:
            for s in SIDES:
                out[f"{s}_{variant}"] = np.nan
            continue
        ra, pa = e["rush_att"].to_numpy(), e["pass_att"].to_numpy()
        if vegas:
            parts = [ctx.prev_resid] if ctx.prev_resid is not None else []
            if len(cur):
                parts.append(_volume_residuals(cur, fixtures, fits[base]))
            empty = pd.DataFrame({"fav": [], "rush_resid": [], "pass_resid": [], "rush_exp": [], "pass_exp": []})
            adj = fit_volume_lines(pd.concat(parts, ignore_index=True) if parts else empty, cfg.lines_mode)
            out.attrs["lines"][base] = adj
            pairs = [adj.apply(a, b, f) for a, b, f in zip(ra, pa, out.fav.fillna(0.0))]
            ra, pa = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        out[f"rush_{variant}"], out[f"pass_{variant}"] = ra * e["rush_ypc"].to_numpy(), pa * e["pass_ypa"].to_numpy()
    return out
