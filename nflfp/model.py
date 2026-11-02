"""Opponent-adjusted expected fantasy points.

Model (per component c, e.g. rushing yards):

    expected[c](offense o vs defense d) = mu[c] * OFF[c][o] * DEF[c][d]

* mu   - league-average output per team-game
* DEF  - how much more/less than average a defense gives up to an average offense
* OFF  - how much more/less than average an offense produces against an average defense

OFF and DEF are fitted by alternating updates (each team's factor is the mean of its
game ratios after removing the strength of the opposition) until they stop changing.
A little shrinkage toward 1.0 (pseudo-games) keeps small samples, especially TDs, sane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# component -> column in the nflverse player-week table
COMPONENTS = {
    "rush_yds": "rushing_yards",
    "rush_tds": "rushing_tds",
    "rec_yds": "receiving_yards",
    "rec_tds": "receiving_tds",
    "receptions": "receptions",
}

# pseudo-games of "average" mixed into every team's factor
DEFAULT_SHRINK = {"rush_yds": 1.0, "rec_yds": 1.0, "receptions": 1.0, "rush_tds": 4.0, "rec_tds": 4.0}


@dataclass
class Scoring:
    ppr: float = 0.5        # per reception
    yard: float = 0.1       # per rushing/receiving yard
    td: float = 6.0         # per rushing/receiving TD
    pass_yard: float = 0.04  # per passing yard
    pass_td: float = 4.0    # per passing TD


@dataclass
class Ratings:
    mu: pd.Series          # index: component
    off: pd.DataFrame      # index: team, columns: component
    dfn: pd.DataFrame      # index: team, columns: component
    raw_allowed: pd.DataFrame  # per-game allowed, unadjusted


def team_games(player_week: pd.DataFrame) -> pd.DataFrame:
    """One row per (offense team, opponent, week) with summed component totals."""
    cols = list(COMPONENTS.values())
    tg = (
        player_week.groupby(["team", "opponent_team", "week"], as_index=False)[cols]
        .sum()
        .rename(columns={v: k for k, v in COMPONENTS.items()})
    )
    return tg.rename(columns={"team": "off_team", "opponent_team": "def_team"})


def _mean_ratio(idx, ratios, counts, shrink):
    """Per-team mean of game ratios with `shrink` pseudo-games of 1.0; teams with no games get 1.0."""
    total = np.bincount(idx, ratios, len(counts)) + shrink
    n = counts + shrink
    return np.divide(total, n, out=np.ones(len(counts)), where=n > 0)


def _fit_component(o, d, y, n_teams, shrink, max_iter, tol):
    mu = y.mean()
    off = np.ones(n_teams)
    dfn = np.ones(n_teams)
    if mu <= 0:
        return mu, off, dfn
    n_o = np.bincount(o, minlength=n_teams)
    n_d = np.bincount(d, minlength=n_teams)
    for _ in range(max_iter):
        new_dfn = _mean_ratio(d, y / (mu * off[o]), n_d, shrink)
        new_dfn /= new_dfn.mean()
        new_off = _mean_ratio(o, y / (mu * new_dfn[d]), n_o, shrink)
        new_off /= new_off.mean()
        delta = max(np.abs(new_dfn - dfn).max(), np.abs(new_off - off).max())
        dfn, off = new_dfn, new_off
        if delta < tol:
            break
    return mu, off, dfn


def fit_ratings(tg: pd.DataFrame, shrink: dict | None = None, max_iter: int = 1000, tol: float = 1e-10) -> Ratings:
    shrink = {**DEFAULT_SHRINK, **(shrink or {})}
    teams = sorted(set(tg["off_team"]) | set(tg["def_team"]))
    ix = {t: i for i, t in enumerate(teams)}
    o = tg["off_team"].map(ix).to_numpy()
    d = tg["def_team"].map(ix).to_numpy()
    mus, offs, dfns = {}, {}, {}
    for c in COMPONENTS:
        mus[c], offs[c], dfns[c] = _fit_component(o, d, tg[c].to_numpy(float), len(teams), shrink[c], max_iter, tol)
    raw = tg.groupby("def_team")[list(COMPONENTS)].mean().reindex(teams)
    return Ratings(
        mu=pd.Series(mus),
        off=pd.DataFrame(offs, index=teams),
        dfn=pd.DataFrame(dfns, index=teams),
        raw_allowed=raw,
    )


def points(comp: pd.Series | pd.DataFrame, s: Scoring):
    """Convert component totals into (rushing points, receiving points)."""
    rush = comp["rush_yds"] * s.yard + comp["rush_tds"] * s.td
    rec = comp["rec_yds"] * s.yard + comp["rec_tds"] * s.td + comp["receptions"] * s.ppr
    return rush, rec


def pass_points(comp, s: Scoring):
    """Quarterback points for a team's passing output (team receiving yards/TDs == passing yards/TDs)."""
    return comp["rec_yds"] * s.pass_yard + comp["rec_tds"] * s.pass_td


def defense_table(r: Ratings, s: Scoring) -> pd.DataFrame:
    """Per defense: raw allowed vs. what they'd allow to an average offense, in fantasy points."""
    adj = r.dfn * r.mu  # allowed to an average offense, per component
    raw_rush, raw_rec = points(r.raw_allowed, s)
    adj_rush, adj_rec = points(adj, s)
    out = pd.DataFrame(
        {
            "raw_rush_pts": raw_rush, "adj_rush_pts": adj_rush,
            "raw_pass_pts": raw_rec, "adj_pass_pts": adj_rec,
        }
    )
    out["adj_total_pts"] = out["adj_rush_pts"] + out["adj_pass_pts"]
    return out.sort_values("adj_total_pts", ascending=False)


def team_expectation(r: Ratings, offense: str, defense: str, use_offense_adj: bool = True) -> pd.Series:
    """Expected component totals for `offense` against `defense`."""
    f = r.mu * r.dfn.loc[defense]
    return f * r.off.loc[offense] if use_offense_adj else f


def usage_shares(player_week: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Each player's share of his *current* team's carries and targets.

    Uses games played for the player's current team (latest team in the data), optionally
    only the last `window` weeks the team played. Shares are per-game-average based, so a
    player who missed games is naturally diluted, and each team's shares sum to 1.
    """
    pw = player_week.sort_values("week")
    cur = pw.groupby("player_id")["team"].last().rename("cur_team")
    pw = pw.join(cur, on="player_id")
    pw = pw[pw["team"] == pw["cur_team"]]
    if window:
        # weeks are not contiguous across byes; use the team's last `window` distinct weeks
        rank = pw.groupby("team")["week"].rank(method="dense", ascending=False)
        pw = pw[rank <= window]
    g = pw.groupby(["cur_team", "player_id"]).agg(
        player=("player_display_name", "last"),
        position=("position", "last"),
        games=("week", "nunique"),
        carries=("carries", "sum"),
        targets=("targets", "sum"),
        attempts=("attempts", "sum"),
    ).reset_index()
    g["carry_share"] = g["carries"] / g.groupby("cur_team")["carries"].transform("sum")
    # all passing points go to the starter: the QB with most attempts in the team's latest game
    latest = pw[pw["week"] == pw.groupby("team")["week"].transform("max")]
    starters = latest.loc[latest.groupby("team")["attempts"].idxmax(), ["team", "player_id"]]
    starters = starters[latest.loc[starters.index, "attempts"] > 0]
    g["attempt_share"] = [
        1.0 if (t, pid) in set(zip(starters.team, starters.player_id)) else 0.0
        for t, pid in zip(g["cur_team"], g["player_id"])
    ]
    g["target_share"] = g["targets"] / g.groupby("cur_team")["targets"].transform("sum")
    return g.rename(columns={"cur_team": "team"}).fillna({"carry_share": 0.0, "target_share": 0.0, "attempt_share": 0.0})


def project_players(
    r: Ratings, shares: pd.DataFrame, matchups: list[tuple[str, str]], s: Scoring, use_offense_adj: bool = True
) -> pd.DataFrame:
    """Split each offense's expected rushing/receiving points among its players."""
    rows = []
    for offense, defense in matchups:
        exp = team_expectation(r, offense, defense, use_offense_adj)
        team_rush, team_rec = points(exp, s)
        team_pass = pass_points(exp, s)
        sh = shares[shares["team"] == offense]
        for p in sh.itertuples():
            rush = p.carry_share * team_rush
            rec = p.target_share * team_rec
            qb = p.attempt_share * team_pass
            if rush == 0 and rec == 0 and qb == 0:
                continue
            rows.append(
                dict(player=p.player, position=p.position, team=offense, opp=defense, games=p.games,
                     carry_share=p.carry_share, target_share=p.target_share,
                     rush_pts=rush, rec_pts=rec, pass_pts=qb, exp_pts=rush + rec + qb)
            )
    cols = ["player", "position", "team", "opp", "games", "carry_share", "target_share", "rush_pts", "rec_pts", "pass_pts", "exp_pts"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("exp_pts", ascending=False).reset_index(drop=True)
