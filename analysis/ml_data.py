"""Feature table for the multivariate player model: one row per player-game (QB/RB/WR/TE who played), everything known before kickoff.

History never reaches further back than the previous season. Feature groups:
  usage    the player's own recent games (last game, last 3, last 8, recency-weighted, this season, last season): points, carries, targets, yards, TDs,
           shares of team volume, snap share, distance-aware expected TDs and goal-line / red-zone opportunities from play-by-play
  context  Vegas implied total, spread and game total, home, the team's recent volume and yards, what the opponent allows to the position (raw inputs only:
           none of the hand-built modules 1-4 feed this model)
  avail    (optional, `AVAIL`) availability known at kickoff: share of the team's recent carries / targets held by players not playing, QB change

    uv run python -m analysis.ml_data        # builds and caches .cache/ml_features.pkl
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from nflfp import attribution as A, data, sleeper, tds as T
from nflfp.model import Scoring

SEASONS = (2024, 2025, 2026)
POS = ("QB", "RB", "WR", "TE")
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
OUT = Path(".cache/ml_features.pkl")
STATS = ["pts", "pts_notd", "carries", "targets", "receptions", "rush_yds", "rec_yds", "pass_att", "pass_yds", "rush_td", "rec_td", "pass_td", "air_yds",
         "target_share", "air_share", "carry_share", "snap_pct", "xtd_rush", "xtd_rec", "gl_car", "rz_car", "rz_tgt"]
AVAIL = ["vac_carry", "vac_target", "qb_new", "gap_games"]


def raw_table(sc: Scoring = Scoring()) -> pd.DataFrame:
    pw = pd.concat([data.load_player_weeks(s) for s in SEASONS], ignore_index=True)
    pw = pw[pw.position.isin(POS)].copy()
    ren = {"rushing_yards": "rush_yds", "receiving_yards": "rec_yds", "attempts": "pass_att", "passing_yards": "pass_yds", "rushing_tds": "rush_td",
           "receiving_tds": "rec_td", "passing_tds": "pass_td", "receiving_air_yards": "air_yds", "air_yards_share": "air_share", "opponent_team": "opp"}
    d = pw.rename(columns=ren)[["player_id", "player_display_name", "position", "season", "week", "team", "opp", "carries", "targets", "receptions", "rush_yds",
                                "rec_yds", "pass_att", "pass_yds", "rush_td", "rec_td", "pass_td", "air_yds", "target_share", "air_share"]].fillna(
        {c: 0.0 for c in ("carries", "targets", "receptions", "rush_yds", "rec_yds", "pass_att", "pass_yds", "rush_td", "rec_td", "pass_td", "air_yds", "target_share", "air_share")})
    d["pts_notd"] = sc.yard * (d.rush_yds + d.rec_yds) + sc.ppr * d.receptions + sc.pass_yard * d.pass_yds
    d["pts"] = d.pts_notd + sc.td * (d.rush_td + d.rec_td) + sc.pass_td * d.pass_td
    d["carry_share"] = d.carries / d.groupby(["season", "week", "team"]).carries.transform("sum").replace(0, np.nan)
    # play-by-play: expected TDs and where the opportunities were
    plays = T.with_xtd(T.load_plays(SEASONS))
    r = plays[(plays.play_type == "run") & plays.rusher_player_id.notna()]
    r = r.assign(gl=(r.yardline_100 <= 5).astype(float), rz=(r.yardline_100 <= 20).astype(float)).groupby(["season", "week", "rusher_player_id"]).agg(
        xtd_rush=("xtd", "sum"), gl_car=("gl", "sum"), rz_car=("rz", "sum")).reset_index().rename(columns={"rusher_player_id": "player_id"})
    c = plays[(plays.play_type == "pass") & plays.receiver_player_id.notna()]
    c = c.assign(rz=(c.yardline_100 <= 20).astype(float)).groupby(["season", "week", "receiver_player_id"]).agg(
        xtd_rec=("xtd", "sum"), rz_tgt=("rz", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    d = d.merge(r, on=["season", "week", "player_id"], how="left").merge(c, on=["season", "week", "player_id"], how="left")
    d[["xtd_rush", "gl_car", "rz_car", "xtd_rec", "rz_tgt"]] = d[["xtd_rush", "gl_car", "rz_car", "xtd_rec", "rz_tgt"]].fillna(0.0)
    # snap share (the snap file has its own ids: match on name + team + week)
    snaps = []
    for s in SEASONS:
        ttl = None if s < data.current_season() else data.CACHE_TTL_HOURS
        x = pd.read_csv(data._fetch(SNAP_URL.format(season=s), f"snaps_{s}.csv", False, ttl))
        snaps.append(x[(x.game_type == "REG") & x.position.isin(POS)])
    sn = pd.concat(snaps)
    sn["key"] = sn.player.map(sleeper.norm)
    sn = sn.groupby(["season", "week", "team", "key"], as_index=False).offense_pct.max().rename(columns={"offense_pct": "snap_pct"})
    d["key"] = d.player_display_name.map(sleeper.norm)
    d = d.merge(sn, on=["season", "week", "team", "key"], how="left").drop(columns="key")
    return d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _player_features(d: pd.DataFrame, s: int) -> pd.DataFrame:
    """Rolling features for the rows of season s, from seasons s-1 and s only."""
    x = d[d.season.isin([s - 1, s])].sort_values(["player_id", "season", "week"]).copy()
    g = x.groupby("player_id", sort=False)
    out = x[["player_id", "season", "week"]].copy()
    for c in STATS:
        prev = g[c].shift(1)
        pg = prev.groupby(x.player_id, sort=False)
        out[f"{c}_l1"] = prev
        out[f"{c}_m3"] = pg.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{c}_m8"] = pg.rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{c}_ew"] = pg.transform(lambda v: v.ewm(halflife=3, ignore_na=True).mean())
    cur = x[x.season == s]
    gc = cur.groupby("player_id", sort=False)
    for c in ("pts", "carries", "targets", "snap_pct", "carry_share", "target_share"):
        out.loc[cur.index, f"{c}_cur"] = gc[c].transform(lambda v: v.shift(1).expanding().mean())
        pm = x[x.season == s - 1].groupby("player_id")[c].mean()
        out[f"{c}_prev"] = x.player_id.map(pm)
    out["n_cur"] = 0.0
    out.loc[cur.index, "n_cur"] = gc.cumcount().astype(float)
    out["n_hist"] = g.cumcount().clip(upper=16).astype(float)
    out["n_prev"] = x.player_id.map(x[x.season == s - 1].groupby("player_id").size()).fillna(0.0)
    return out[out.season == s]


def _team_context(d: pd.DataFrame, s: int) -> pd.DataFrame:
    """Team's recent volume and what the opponent has allowed to the position (last 8 games, seasons s-1 and s)."""
    x = d[d.season.isin([s - 1, s])]
    tg = x.groupby(["season", "week", "team"], as_index=False).agg(tm_att=("pass_att", "sum"), tm_car=("carries", "sum"), tm_pts=("pts", "sum"), tm_pyd=("pass_yds", "sum"), tm_ryd=("rush_yds", "sum")).sort_values(["team", "season", "week"])
    for c in ("tm_att", "tm_car", "tm_pts", "tm_pyd", "tm_ryd"):
        tg[f"{c}_m5"] = tg.groupby("team")[c].transform(lambda v: v.shift(1).rolling(5, min_periods=1).mean())
    al = x.groupby(["season", "week", "opp", "position"], as_index=False).pts.sum().sort_values(["opp", "position", "season", "week"])
    al["def_pos_m8"] = al.groupby(["opp", "position"]).pts.transform(lambda v: v.shift(1).rolling(8, min_periods=2).mean())
    lg = al.groupby(["season", "week", "position"]).pts.mean().groupby("position").transform(lambda v: v.shift(1).expanding().mean()).rename("lg_pos").reset_index()
    al = al.merge(lg, on=["season", "week", "position"], how="left")
    al["def_pos_rel"] = al.def_pos_m8 / al.lg_pos
    rows = x[x.season == s][["player_id", "season", "week", "team", "opp", "position"]]
    rows = rows.merge(tg[["season", "week", "team", "tm_att_m5", "tm_car_m5", "tm_pts_m5", "tm_pyd_m5", "tm_ryd_m5"]], on=["season", "week", "team"], how="left")
    return rows.merge(al[["season", "week", "opp", "position", "def_pos_m8", "def_pos_rel"]], on=["season", "week", "opp", "position"], how="left")


def _availability(d: pd.DataFrame, s: int) -> pd.DataFrame:
    """Per player-row of season s: share of the team's recent carries / targets held by players who are not playing this game, QB change, and the
    player's own missed team games since he last played. Uses who actually played (known at kickoff from the inactive list)."""
    x = d[d.season == s].sort_values(["player_id", "week"]).copy()
    for c in ("carry_share", "target_share"):
        x[f"{c}_post"] = x.groupby("player_id")[c].transform(lambda v: v.fillna(0).ewm(halflife=3).mean())      # after the game: what a later week knows
    recs = []
    for team, tg in x.groupby("team"):
        weeks = sorted(tg.week.unique())
        for i, w in enumerate(weeks):
            past = weeks[max(0, i - 3):i]
            active = set(tg[tg.week == w].player_id)
            rec = dict(season=s, week=w, team=team, vac_carry=0.0, vac_target=0.0, qb_new=np.nan)
            if past:
                h = tg[tg.week.isin(past)].sort_values("week").groupby("player_id").last()
                out = h[~h.index.isin(active)]
                rec["vac_carry"], rec["vac_target"] = float(out.carry_share_post.sum()), float(out.target_share_post.sum())
                qb = lambda wk: tg[tg.week == wk].sort_values("pass_att").player_id.iloc[-1]
                rec["qb_new"] = float(qb(w) != qb(past[-1]))
            recs.append(rec)
    team_av = pd.DataFrame(recs)
    tgi = x[["team", "week"]].drop_duplicates().sort_values(["team", "week"])
    tgi["tgame"] = tgi.groupby("team").cumcount()
    x = x.merge(tgi, on=["team", "week"])
    x = x.sort_values(["player_id", "week"])
    same = x.groupby("player_id").team.shift(1) == x.team
    x["gap_games"] = (x.tgame - x.groupby("player_id").tgame.shift(1) - 1).where(same)
    return x[["player_id", "season", "week", "team", "gap_games"]].merge(team_av, on=["season", "week", "team"], how="left")


def sleeper_points(season: int, week: int, known: pd.DataFrame, dump: dict, sc: Scoring = Scoring()) -> pd.DataFrame:
    """Sleeper's pre-game projection in our scoring for every player it projects that week (cached)."""
    cache = Path(f".cache/sleeper_proj_{season}_{week}.pkl")
    if cache.exists() and (season < data.current_season() or time.time() - cache.stat().st_mtime < 6 * 3600):
        v = pd.read_pickle(cache)
    else:
        v = A.sleeper_volume(season, week)
        v.to_pickle(cache)
    v = v.copy()
    v["player_id"] = v.sleeper_id.map(sleeper.map_ids(dump, known, v.sleeper_id))
    v["sl"] = (sc.yard * (v.sl_rush_yds + v.sl_rec_yds) + sc.ppr * v.sl_receptions + sc.pass_yard * v.sl_pass_yds
               + sc.td * (v.sl_rush_tds + v.sl_rec_tds) + sc.pass_td * v.sl_pass_tds)
    v["season"], v["week"] = season, week
    return v[v.player_id.notna()][["player_id", "season", "week", "sl", "carries", "targets", "attempts"]].rename(
        columns={"carries": "sl_carries", "targets": "sl_targets", "attempts": "sl_attempts"}).drop_duplicates(["player_id"])


def build() -> pd.DataFrame:
    d = raw_table()
    sched = pd.read_csv(".cache/games.csv", low_memory=False)
    sched = sched[sched.season.isin(SEASONS) & (sched.game_type == "REG")]
    h = sched.assign(team=sched.home_team, home=1.0, fav=sched.spread_line, implied=sched.total_line / 2 + sched.spread_line / 2)
    a = sched.assign(team=sched.away_team, home=0.0, fav=-sched.spread_line, implied=sched.total_line / 2 - sched.spread_line / 2)
    veg = pd.concat([h, a])[["season", "week", "team", "home", "fav", "implied", "total_line"]]
    dump = sleeper.load_players()
    parts = []
    for s in SEASONS[1:]:
        base = d[d.season == s][["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts", "pts_notd"]]
        f = base.merge(_player_features(d, s), on=["player_id", "season", "week"], how="left")
        f = f.merge(_team_context(d, s).drop(columns=["team", "opp", "position"]), on=["player_id", "season", "week"], how="left")
        f = f.merge(_availability(d, s).drop(columns="team"), on=["player_id", "season", "week"], how="left")
        f = f.merge(veg, on=["season", "week", "team"], how="left")
        known = d[d.season == s].sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
        sl = pd.concat([sleeper_points(s, int(w), known, dump) for w in sorted(base.week.unique())], ignore_index=True)
        parts.append(f.merge(sl, on=["player_id", "season", "week"], how="left"))
    F = pd.concat(parts, ignore_index=True)
    for p in POS:
        F[f"is_{p}"] = (F.position == p).astype(float)
    return F


if __name__ == "__main__":
    F = build()
    F.to_pickle(OUT)
    print(F.shape, "rows by season:", F.groupby("season").size().to_dict())
    print("snap share present:", F.snap_pct_l1.notna().mean().round(3), "| Sleeper projection present:", F.sl.notna().mean().round(3),
          "| implied present:", F.implied.notna().mean().round(3))
    print(F[F.player_display_name == "Chase Brown"].tail(3)[["season", "week", "pts", "pts_l1", "pts_m3", "carries_ew", "snap_pct_m3", "xtd_rush_m8", "implied", "vac_carry", "sl"]])
