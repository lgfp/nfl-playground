"""Module 1 - expected team rushing and passing yards for a week (the final design).

Two chains, each built one backtested step at a time (see analysis/ablation.py, stack.py, rush_ridge.py):

    rush = ridge average  ->  Vegas
    pass = ridge opponent-adjusted fit  ->  wind  ->  Vegas

* ridge average (rush): the team's average rush yards this season so far, pulled toward a target by `rush_strength` pseudo-games
      estimate = (n * team average + K * target) / (n + K),   target = w * last season's team average + (1 - w) * league average
  with K = 12 and w = 0.15 by default (n = the team's games so far; with no games the estimate is the target).
* ridge opponent-adjusted fit (pass): the iterative multiplicative model  MU * OFF[team] * DEF[opponent]  of nflfp.yards, whose factors are
  pulled toward 0.3 * last season's team factor + 0.7 * 1.0 with 8 pseudo-games.
* wind (pass): + (passing yards per 5 mph) * (game wind - average wind); indoors counts as 0 wind; unknown wind is treated as average.
* Vegas (both): + (yards per point favoured) * spread, a separate slope for rush and pass (favourites gain both).

Every slope is a least-squares fit through the origin on the residuals of the stage before it, over last season plus this season's earlier
weeks, with each of those rows predicted as it would have been at the time (walk-forward). Only earlier games are ever used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import lines as linemod
from . import yards as Y

KEYS = ["season", "week", "off_team", "def_team"]


@dataclass(frozen=True)
class TeamYardsConfig:
    rush_strength: float = 12.0        # pseudo-games of the target in the rush average
    rush_prior_weight: float = 0.15    # weight of last season's team average in the rush target (rest: league average)
    pass_strength: float = 8.0         # pseudo-games in the pass opponent-adjusted fit
    pass_prior_weight: float = 0.3     # weight of last season's team factor in the pass target (rest: league average)


# --------------------------------------------------------------------------------------------------- inputs
def environment(schedule: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (season, week, offense, defense) with `fav` (points the offense is favoured by) and `wind5` (wind in 5 mph units; 0
    indoors, NaN if unknown). `schedule` is nflverse games for one season; `weather` is weather.game_weather(schedule) (index game_id)."""
    fx = Y.fixtures_from_schedule(schedule)
    if weather is None:
        return fx.assign(wind5=np.nan)
    tl = linemod.team_lines(schedule, weather).rename(columns={"team": "off_team"})[["week", "off_team", "wind5"]]
    return fx.merge(tl.drop_duplicates(["week", "off_team"]), on=["week", "off_team"], how="left")


def _slope(x: np.ndarray, res: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Least-squares slope through the origin; rows with a missing value are skipped."""
    m = np.isfinite(x) & np.isfinite(res) if mask is None else mask & np.isfinite(x) & np.isfinite(res)
    den = float((x[m] ** 2).sum())
    return float((x[m] * res[m]).sum() / den) if den else 0.0


# -------------------------------------------------------------------------------------------- the engine
def walk_forward(games: pd.DataFrame, env: pd.DataFrame, seasons: list[int], cfg: TeamYardsConfig = TeamYardsConfig(),
                 last_week: int | None = None) -> pd.DataFrame:
    """Expected rush / pass yards for every fixture of `seasons`, each week predicted from earlier weeks only.

    games: Y.team_game_yards() output (season, week, off_team, def_team, rush_yds, pass_yds), every season needed (the previous season of
    the first one too, for its prior); env: environment() output for the seasons. Fixtures without a game yet (future weeks) are projected too.
    last_week stops the last season after that week.
    """
    G = env.merge(games[KEYS + ["rush_yds", "pass_yds"]], on=KEYS, how="left")
    teams = sorted(set(env.off_team) | set(env.def_team) | set(games.off_team))
    ycfg = Y.YardsConfig(ridge_strength=cfg.pass_strength, prior_weight=cfg.pass_prior_weight, prior_basis="raw")
    hist, out = [], []
    for season in seasons:
        prev = games[games.season == season - 1]
        prev_mu = {s: float(prev[f"{s}_yds"].mean()) for s in Y.SIDES} if len(prev) else None
        prev_rush = prev.groupby("off_team").rush_yds.mean() if len(prev) else pd.Series(dtype=float)
        prior = Y.build_prior(prev[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, ycfg) if len(prev) else None
        for week in sorted(G.loc[G.season == season, "week"].unique()):
            if last_week is not None and season == seasons[-1] and week > last_week:
                break
            cur = games[(games.season == season) & (games.week < week)]
            todo = G[(G.season == season) & (G.week == week)].copy()
            mu = {s: float(cur[f"{s}_yds"].mean()) if len(cur) else (prev_mu[s] if prev_mu else np.nan) for s in Y.SIDES}
            if np.isnan(mu["rush"]):                                  # nothing to go on (first week of the first season)
                continue
            # -- rush: ridge average
            n = todo.off_team.map(cur.groupby("off_team").size()).fillna(0.0)
            own = todo.off_team.map(cur.groupby("off_team").rush_yds.mean()).fillna(mu["rush"])
            last = todo.off_team.map(prev_rush).fillna(mu["rush"])
            target = cfg.rush_prior_weight * last + (1 - cfg.rush_prior_weight) * mu["rush"]
            todo["rush_ridge"] = ((n * own + cfg.rush_strength * target) / (n + cfg.rush_strength)).to_numpy(float)
            # -- pass: ridge opponent-adjusted fit
            fit = Y.fit_ratings(cur[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, ycfg, prior, ridge=True, fallback_mu=prev_mu)
            todo["pass_ridge_opp"] = [fit.expected(o, d)["pass"] for o, d in zip(todo.off_team, todo.def_team)]
            # -- slopes from last season + this season's earlier weeks, each row as it was predicted then
            tr = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame(columns=["season", "fav", "wind5", "rush_yds", "pass_yds", "rush_ridge", "pass_ridge_opp", "pass_wind"])
            tr = tr[tr.season.isin([season - 1, season])]
            fav_tr, w_tr = tr.fav.to_numpy(float), tr.wind5.to_numpy(float)
            wmean = float(np.nanmean(w_tr)) if np.isfinite(w_tr).any() else 0.0
            s_rush = _slope(fav_tr, (tr.rush_yds - tr.rush_ridge).to_numpy(float))
            s_wind = _slope(w_tr - wmean, (tr.pass_yds - tr.pass_ridge_opp).to_numpy(float))
            s_pass = _slope(fav_tr, (tr.pass_yds - tr.pass_wind).to_numpy(float))
            fav = todo.fav.to_numpy(float)
            todo["rush_exp"] = np.maximum(todo.rush_ridge.to_numpy(float) + s_rush * fav, 0.0)
            todo["pass_wind"] = np.maximum(todo.pass_ridge_opp.to_numpy(float) + s_wind * (todo.wind5.fillna(wmean).to_numpy(float) - wmean), 0.0)
            todo["pass_exp"] = np.maximum(todo.pass_wind.to_numpy(float) + s_pass * fav, 0.0)
            todo["slope_rush_vegas"], todo["slope_pass_wind"], todo["slope_pass_vegas"] = s_rush, s_wind, s_pass
            out.append(todo)
            played = todo[todo.rush_yds.notna()]
            if len(played):
                hist.append(played[["season", "fav", "wind5", "rush_yds", "pass_yds", "rush_ridge", "pass_ridge_opp", "pass_wind"]])
    cols = KEYS + ["fav", "wind5", "rush_ridge", "rush_exp", "pass_ridge_opp", "pass_wind", "pass_exp",
                   "slope_rush_vegas", "slope_pass_wind", "slope_pass_vegas", "rush_yds", "pass_yds"]
    return pd.concat(out, ignore_index=True)[cols]


def project_week(games: pd.DataFrame, env: pd.DataFrame, season: int, week: int, cfg: TeamYardsConfig = TeamYardsConfig()) -> pd.DataFrame:
    """Expected rush and pass yards (`rush_exp`, `pass_exp`) for every fixture of one week, from earlier games only.

    Also returns the stage-by-stage values (rush_ridge; pass_ridge_opp, pass_wind) and the fitted slopes used.
    """
    out = walk_forward(games, env, [season - 1, season], cfg, last_week=week)
    return out[(out.season == season) & (out.week == week)].reset_index(drop=True)
