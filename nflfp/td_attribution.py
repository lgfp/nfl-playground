"""Module 4 - split module 3's team touchdowns among the players.

Each player's share comes from his team's play-by-play history, the team's games since the previous season weighted `decay ** age` (age = team games ago, so a
bye does not count as a game):

    rushing TDs   = team rush TDs  x  player's share of the team's rushing xTD    (every carry weighted by the league scoring rate at its yard line)
    receiving TDs = team pass TDs  x  player's share of the team's targets
    passing TDs   = team pass TDs  x  QB's share of projected pass attempts       (module 2's Sleeper attempts share)

Backtest (analysis/td_player.py, 2025-26 weeks 4+): the xTD share beats carry share for rushing (about 13% vs 12% lower squared error than the average
candidate; larger for the top backs), while for receiving plain target share is the best. A player's own TD history, red-zone share and fitted combinations add nothing.

Players in `exclude` (Out / IR) are removed from both the player and the team totals, so their share goes to teammates in proportion to the teammates' own.
A player with no games for his current team in that history (rookie, trade) gets no touchdowns here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Scoring


@dataclass(frozen=True)
class ShareConfig:
    decay: float = 0.85      # weight of a team game k games back = decay ** k


def _weighted(plays: pd.DataFrame, kind: str, season: int, week: int, decay: float) -> pd.DataFrame:
    """Per (team, player): decay-weighted xTD (rush) or targets (rec) over the team's games before (season, week), season - 1 at the earliest."""
    run = kind == "rush"
    p = plays[(plays.play_type == ("run" if run else "pass")) & plays[("rusher" if run else "receiver") + "_player_id"].notna()]
    p = p[(p.season >= season - 1) & ((p.season < season) | (p.week < week))]
    games = p[["posteam", "season", "week"]].drop_duplicates().sort_values(["posteam", "season", "week"], ascending=[True, False, False])
    games["age"] = games.groupby("posteam").cumcount()
    p = p.merge(games, on=["posteam", "season", "week"])
    p["w"] = np.power(decay, p.age.to_numpy(float)) * (p.xtd if run else 1.0)
    pid, name = ("rusher_player_id", "rusher_player_name") if run else ("receiver_player_id", "receiver_player_name")
    g = p.groupby(["posteam", pid], as_index=False).agg(name=(name, "last"), w=("w", "sum"))
    return g.rename(columns={"posteam": "team", pid: "pid"})


def shares(plays: pd.DataFrame, season: int, week: int, exclude: set | frozenset = frozenset(), cfg: ShareConfig = ShareConfig()) -> pd.DataFrame:
    """Each player's rush_share (of the team's rushing xTD) and rec_share (of its targets) for `week`, from games before it.

    plays: tds.with_xtd(tds.load_plays(...)) covering season - 1 and season. Shares of a team sum to 1 over the players not in `exclude`.
    """
    out = None
    for kind, col in (("rush", "rush_share"), ("rec", "rec_share")):
        g = _weighted(plays, kind, season, week, cfg.decay)
        g = g[~g.pid.isin(exclude)]
        tot = g.groupby("team").w.transform("sum")
        g[col] = (g.w / tot).where(tot > 0, 0.0)
        g = g[["team", "pid", "name", col]]
        out = g if out is None else out.merge(g, on=["team", "pid"], how="outer", suffixes=("", "_r"))
        if "name_r" in out:
            out["name"] = out.name.fillna(out.name_r)
            out = out.drop(columns="name_r")
    return out.fillna({"rush_share": 0.0, "rec_share": 0.0}).reset_index(drop=True)


def attribute(team_tds: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    """Expected rushing and receiving TDs per player. team_tds: module 3 rows (off_team, def_team, rush_td_exp, pass_td_exp); sh: shares() output."""
    t = team_tds[["off_team", "def_team", "rush_td_exp", "pass_td_exp"]].rename(columns={"off_team": "team", "def_team": "opp"})
    d = sh.merge(t, on="team", how="inner")
    d["rush_tds"] = d.rush_share * d.rush_td_exp
    d["rec_tds"] = d.rec_share * d.pass_td_exp
    return d[["team", "opp", "pid", "name", "rush_share", "rec_share", "rush_tds", "rec_tds", "rush_td_exp", "pass_td_exp"]]


def add_touchdowns(players: pd.DataFrame, td: pd.DataFrame, id_map: dict, scoring: Scoring = Scoring()) -> pd.DataFrame:
    """Adds TD columns to module 2's player table.

    players: attribution.attribute() output (needs sleeper_id, team, attempts_share, exp_pts); td: attribute() output above;
    id_map: {sleeper id: nflverse id} for the players (sleeper.map_ids). A player without a match, or without history for his team, gets 0 rush / rec TDs.
    Returns the table with rush_tds, rec_tds, pass_tds (QB: team pass TDs x his attempts share), td_pts and total_pts (module 2's points + TD points).
    """
    d = players.copy()
    d["pid"] = d.sleeper_id.map(id_map)
    t = td.drop_duplicates(["team", "pid"]).set_index(["team", "pid"])
    key = list(zip(d.team, d.pid))
    d["rush_tds"] = [float(t.rush_tds.get(k, 0.0)) for k in key]
    d["rec_tds"] = [float(t.rec_tds.get(k, 0.0)) for k in key]
    team_pass = td.drop_duplicates("team").set_index("team").pass_td_exp
    d["pass_tds"] = d.attempts_share * d.team.map(team_pass).fillna(0.0)
    d["td_pts"] = scoring.td * (d.rush_tds + d.rec_tds) + scoring.pass_td * d.pass_tds
    d["total_pts"] = d.exp_pts + d.td_pts
    return d.drop(columns="pid")
