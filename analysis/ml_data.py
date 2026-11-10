"""Feature table for the multivariate player model: one row per player-game (QB/RB/WR/TE who played), everything known before kickoff.

History never reaches further back than the previous season. Feature groups:
  usage    the player's own recent games (last game, last 3, last 8, recency-weighted, this season, last season): points, carries, targets, yards, TDs,
           shares of team volume, snap share, distance-aware expected TDs and goal-line / red-zone opportunities from play-by-play
  context  Vegas implied total, spread and game total, home, the team's recent volume and yards, what the opponent allows to the position (raw inputs only:
           none of the hand-built modules 1-4 feed this model)
  avail    (optional, `AVAIL`) availability known at kickoff: share of the team's recent carries / targets held by players not playing, QB change
  extra    (optional, by column prefix, see GROUPS) px_ finer player detail, off_ team offense trends, dfn_ opposing defense, env_ game environment

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


PBP_COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type", "two_point_attempt", "qb_kneel", "qb_spike", "yardline_100", "rush_touchdown",
            "pass_touchdown", "rusher_player_id", "rusher_player_name", "receiver_player_id", "receiver_player_name", "passer_player_id", "epa", "success", "pass", "rush",
            "pass_oe", "qb_dropback", "qb_scramble", "sack", "air_yards", "yards_after_catch", "yards_gained", "down", "score_differential", "qtr", "fixed_drive",
            "game_id", "goal_to_go", "cpoe", "interception"]
# extra per-game player stats (group "px"): efficiency and finer play-by-play detail
PX = ["rush_epa", "rec_epa", "car_in10", "tgt_in10", "ez_tgt", "tgt_deep", "yac", "tgt_3rd", "scrambles", "dropbacks", "qb_epa", "cpoe", "sacks_taken", "deep_att", "ints",
      "rz_share", "gl_share", "first_downs"]
TEAM_STATS = ["plays", "pass_rate", "neutral_pass", "proe", "epa", "epa_pass", "epa_rush", "succ", "expl", "rz_trips", "rz_plays", "gtg", "sack_rate", "tds", "points"]
_PBP = None


def pbp() -> pd.DataFrame:
    """Run and pass plays of the seasons (same filter as nflfp.tds.load_plays) with the extra columns and `xtd`."""
    global _PBP
    if _PBP is None:
        parts = []
        for y in SEASONS:
            ttl = None if y < data.current_season() else data.CACHE_TTL_HOURS
            parts.append(pd.read_csv(data._fetch(data.PBP_URL.format(season=y), f"pbp_{y}.csv.gz", False, ttl), usecols=PBP_COLS, low_memory=False))
        d = pd.concat(parts, ignore_index=True)
        d = d[(d.season_type == "REG") & d.posteam.notna() & d.play_type.isin(["run", "pass"]) & (d.two_point_attempt != 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
        d["td"] = np.where(d.play_type == "run", d.rush_touchdown, d.pass_touchdown).astype(float)
        d["bin"] = pd.cut(d.yardline_100, T.EDGES, labels=False).astype(float)
        _PBP = T.with_xtd(d.reset_index(drop=True))
    return _PBP


def raw_table(sc: Scoring = Scoring()) -> pd.DataFrame:
    pw = pd.concat([data.load_player_weeks(s) for s in SEASONS], ignore_index=True)
    pw = pw[pw.position.isin(POS)].copy()
    ren = {"rushing_yards": "rush_yds", "receiving_yards": "rec_yds", "attempts": "pass_att", "passing_yards": "pass_yds", "rushing_tds": "rush_td",
           "receiving_tds": "rec_td", "passing_tds": "pass_td", "receiving_air_yards": "air_yds", "air_yards_share": "air_share", "opponent_team": "opp"}
    pw["first_downs"] = pw.rushing_first_downs.fillna(0) + pw.receiving_first_downs.fillna(0)
    d = pw.rename(columns=ren)[["first_downs", "player_id", "player_display_name", "position", "season", "week", "team", "opp", "carries", "targets", "receptions", "rush_yds",
                                "rec_yds", "pass_att", "pass_yds", "rush_td", "rec_td", "pass_td", "air_yds", "target_share", "air_share"]].fillna(
        {c: 0.0 for c in ("carries", "targets", "receptions", "rush_yds", "rec_yds", "pass_att", "pass_yds", "rush_td", "rec_td", "pass_td", "air_yds", "target_share", "air_share")})
    d["pts_notd"] = sc.yard * (d.rush_yds + d.rec_yds) + sc.ppr * d.receptions + sc.pass_yard * d.pass_yds
    d["pts"] = d.pts_notd + sc.td * (d.rush_td + d.rec_td) + sc.pass_td * d.pass_td
    d["carry_share"] = d.carries / d.groupby(["season", "week", "team"]).carries.transform("sum").replace(0, np.nan)
    # play-by-play: expected TDs and where the opportunities were
    plays = pbp()
    k = ["season", "week"]
    r = plays[(plays.play_type == "run") & plays.rusher_player_id.notna()]
    r = r.assign(gl=(r.yardline_100 <= 5).astype(float), rz=(r.yardline_100 <= 20).astype(float), in10=(r.yardline_100 <= 10).astype(float)).groupby(k + ["rusher_player_id"]).agg(
        xtd_rush=("xtd", "sum"), gl_car=("gl", "sum"), rz_car=("rz", "sum"), rush_epa=("epa", "sum"), car_in10=("in10", "sum"), scrambles=("qb_scramble", "sum")
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})
    c = plays[(plays.play_type == "pass") & plays.receiver_player_id.notna()]
    c = c.assign(rz=(c.yardline_100 <= 20).astype(float), in10=(c.yardline_100 <= 10).astype(float), ez=(c.air_yards >= c.yardline_100).astype(float),
                 deep=(c.air_yards >= 20).astype(float), d3=(c.down == 3).astype(float)).groupby(k + ["receiver_player_id"]).agg(
        xtd_rec=("xtd", "sum"), rz_tgt=("rz", "sum"), rec_epa=("epa", "sum"), tgt_in10=("in10", "sum"), ez_tgt=("ez", "sum"), tgt_deep=("deep", "sum"),
        yac=("yards_after_catch", "sum"), tgt_3rd=("d3", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    q = plays[(plays.qb_dropback == 1) & plays.passer_player_id.notna()]
    q = q.assign(deep=(q.air_yards >= 20).astype(float)).groupby(k + ["passer_player_id"]).agg(
        dropbacks=("epa", "size"), qb_epa=("epa", "sum"), cpoe=("cpoe", "mean"), sacks_taken=("sack", "sum"), deep_att=("deep", "sum"), ints=("interception", "sum")
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    d = d.merge(r, on=k + ["player_id"], how="left").merge(c, on=k + ["player_id"], how="left").merge(q, on=k + ["player_id"], how="left")
    zero = ["xtd_rush", "gl_car", "rz_car", "xtd_rec", "rz_tgt", "rush_epa", "car_in10", "scrambles", "rec_epa", "tgt_in10", "ez_tgt", "tgt_deep", "yac", "tgt_3rd",
            "dropbacks", "qb_epa", "sacks_taken", "deep_att", "ints"]
    d[zero] = d[zero].fillna(0.0)
    tk = ["season", "week", "team"]
    rz = d.rz_car + d.rz_tgt
    d["rz_share"] = rz / rz.groupby([d.season, d.week, d.team]).transform("sum").replace(0, np.nan)
    d["gl_share"] = d.gl_car / d.groupby(tk).gl_car.transform("sum").replace(0, np.nan)
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
    for c in PX:                                                        # extra detail: last 8 and recency-weighted only
        prev = g[c].shift(1)
        pg = prev.groupby(x.player_id, sort=False)
        out[f"px_{c}_m8"] = pg.rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"px_{c}_ew"] = pg.transform(lambda v: v.ewm(halflife=3, ignore_na=True).mean())
    div = lambda a, b: a / b.where(b > 0)
    opp = out.carries_m8 + out.targets_m8
    out["px_epa_per_opp"] = div(out.px_rush_epa_m8 + out.px_rec_epa_m8, opp)
    out["px_ypc"], out["px_ypt"] = div(out.rush_yds_m8, out.carries_m8), div(out.rec_yds_m8, out.targets_m8)
    out["px_catch_rate"], out["px_adot"] = div(out.receptions_m8, out.targets_m8), div(out.air_yds_m8, out.targets_m8)
    out["px_yac_per_rec"] = div(out.px_yac_m8, out.receptions_m8)
    out["px_qb_epa_per_db"], out["px_ypa"] = div(out.px_qb_epa_m8, out.px_dropbacks_m8), div(out.pass_yds_m8, out.pass_att_m8)
    out["px_td_over_exp"] = (out.rush_td_m8 + out.rec_td_m8) - (out.xtd_rush_m8 + out.xtd_rec_m8)      # scored more than his chances implied (luck)
    out["px_pts_trend"] = out.pts_m3 - out.pts_m8
    out["px_snap_trend"] = out.snap_pct_m3 - out.snap_pct_m8
    out["px_opp_trend"] = (out.carries_m3 + out.targets_m3) - opp
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


def _team_trends(d: pd.DataFrame, s: int, sched: pd.DataFrame) -> pd.DataFrame:
    """Per player-row of season s: his offense's and the opposing defense's recent play-by-play profile (groups "off_" and "dfn_"), last 4 and last 8 games."""
    p = pbp()
    x = p[p.season.isin([s - 1, s])].copy()
    neutral = (x.score_differential.abs() <= 7) & (x.qtr <= 3)
    x["neutral_pass"] = x["pass"].where(neutral)
    x["epa_pass"], x["epa_rush"] = x.epa.where(x["pass"] == 1), x.epa.where(x["rush"] == 1)
    x["expl"] = (x.yards_gained >= 15).astype(float)
    x["rzp"] = (x.yardline_100 <= 20).astype(float)
    x["sack_rate"] = x.sack.where(x["pass"] == 1)
    pts = pd.concat([sched.assign(team=sched.home_team, points=sched.home_score, allowed=sched.away_score),
                     sched.assign(team=sched.away_team, points=sched.away_score, allowed=sched.home_score)])[["season", "week", "team", "points", "allowed"]]
    out = {}
    for side, key in (("off", "posteam"), ("dfn", "defteam")):
        g = x.groupby(["season", "week", key])
        t = g.agg(plays=("epa", "size"), pass_rate=("pass", "mean"), neutral_pass=("neutral_pass", "mean"), proe=("pass_oe", "mean"), epa=("epa", "mean"),
                  epa_pass=("epa_pass", "mean"), epa_rush=("epa_rush", "mean"), succ=("success", "mean"), expl=("expl", "mean"), rz_plays=("rzp", "sum"),
                  gtg=("goal_to_go", "sum"), sack_rate=("sack_rate", "mean"), tds=("td", "sum")).reset_index().rename(columns={key: "team"})
        trips = x[x.yardline_100 <= 20].groupby(["season", "week", key]).fixed_drive.nunique().rename("rz_trips").reset_index().rename(columns={key: "team"})
        t = t.merge(trips, on=["season", "week", "team"], how="left").fillna({"rz_trips": 0.0})
        t = t.merge(pts.rename(columns={"points": "points"} if side == "off" else {"allowed": "points", "points": "_scored"}), on=["season", "week", "team"], how="left")
        t = t.sort_values(["team", "season", "week"])
        f = t[["season", "week", "team"]].copy()
        for c in TEAM_STATS:
            sh = t.groupby("team")[c].shift(1)
            f[f"{side}_{c}_m4"] = sh.groupby(t.team).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
            f[f"{side}_{c}_m8"] = sh.groupby(t.team).rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)
        for c in ("pass_rate", "plays", "epa", "points"):
            f[f"{side}_{c}_trend"] = f[f"{side}_{c}_m4"] - f[f"{side}_{c}_m8"]
        out[side] = f[f.season == s]
    rows = d[d.season == s][["player_id", "season", "week", "team", "opp"]]
    rows = rows.merge(out["off"], on=["season", "week", "team"], how="left")
    return rows.merge(out["dfn"].rename(columns={"team": "opp"}), on=["season", "week", "opp"], how="left").drop(columns=["team", "opp"])


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
    dome = sched.roof.isin(["dome", "closed"]).astype(float)
    h = sched.assign(team=sched.home_team, home=1.0, fav=sched.spread_line, implied=sched.total_line / 2 + sched.spread_line / 2, env_rest=sched.home_rest, env_dome=dome)
    a = sched.assign(team=sched.away_team, home=0.0, fav=-sched.spread_line, implied=sched.total_line / 2 - sched.spread_line / 2, env_rest=sched.away_rest, env_dome=dome)
    veg = pd.concat([h, a]).rename(columns={"div_game": "env_div"})
    veg["env_opp_implied"] = veg.total_line - veg.implied
    veg["env_grass"] = veg.surface.astype(str).str.contains("grass").astype(float)
    veg = veg[["season", "week", "team", "home", "fav", "implied", "total_line", "env_rest", "env_dome", "env_div", "env_opp_implied", "env_grass"]]
    dump = sleeper.load_players()
    parts = []
    for s in SEASONS[1:]:
        base = d[d.season == s][["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts", "pts_notd"]]
        f = base.merge(_player_features(d, s), on=["player_id", "season", "week"], how="left")
        f = f.merge(_team_context(d, s).drop(columns=["team", "opp", "position"]), on=["player_id", "season", "week"], how="left")
        f = f.merge(_availability(d, s).drop(columns="team"), on=["player_id", "season", "week"], how="left")
        f = f.merge(_team_trends(d, s, sched), on=["player_id", "season", "week"], how="left")
        f = f.merge(veg, on=["season", "week", "team"], how="left")
        known = d[d.season == s].sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
        sl = pd.concat([sleeper_points(s, int(w), known, dump) for w in sorted(base.week.unique())], ignore_index=True)
        parts.append(f.merge(sl, on=["player_id", "season", "week"], how="left"))
    F = pd.concat(parts, ignore_index=True)
    return pd.concat([F, pd.DataFrame({f"is_{p}": (F.position == p).astype(float) for p in POS})], axis=1)


GROUPS = {"px": "player efficiency and play-by-play detail (EPA, depth, inside-10, shares, trends)", "off": "team offense trends (run/pass mix, EPA, red-zone trips, points)",
          "dfn": "opposing defense allowed (EPA, red-zone trips, points, pass rate faced)", "env": "game environment (rest, dome, surface, division, opponent implied total)"}


if __name__ == "__main__":
    F = build()
    F.to_pickle(OUT)
    print(F.shape, "rows by season:", F.groupby("season").size().to_dict())
    print("snap share present:", F.snap_pct_l1.notna().mean().round(3), "| Sleeper projection present:", F.sl.notna().mean().round(3),
          "| implied present:", F.implied.notna().mean().round(3))
    print(F[F.player_display_name == "Chase Brown"].tail(3)[["season", "week", "pts", "pts_l1", "pts_m3", "carries_ew", "snap_pct_m3", "xtd_rush_m8", "implied", "vac_carry", "sl"]])
