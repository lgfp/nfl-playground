"""Every model we tested, run on one upcoming week for a Sleeper user's roster.

For a week that has not been played, the feature table has no rows, so pre-game rows are created: one per player Sleeper projects who is not ruled out (Out / IR),
with the opponent and kickoff date from the schedule and the team's likely QB taken from Sleeper's projected attempts. Every input is then computed exactly as for a
played game, from earlier games only. Models are trained on 2021-25, as in the backtests.

    uv run python -m analysis.ml_week <username> [week]
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
import pandas as pd

from analysis import ml_data as MD
from analysis.ml_model import SL_FEATS, feature_sets, fit_predict, gbm
from nflfp import attribution as A, data, league, sleeper

warnings.filterwarnings("ignore")
SEASON = 2026


def map_to_nflverse(vol: pd.DataFrame, known: pd.DataFrame, dump: dict) -> pd.Series:
    """Sleeper id -> nflverse player_id: Sleeper's gsis id / name + team + position first, then name + position when that is unambiguous."""
    pid = vol.sleeper_id.map(sleeper.map_ids(dump, known, vol.sleeper_id))
    k = known.assign(key=known.player_display_name.map(sleeper.norm))
    uniq = k.groupby(["key", "position"]).player_id.agg(["first", "size"])
    uniq = uniq[uniq["size"] == 1]["first"]
    miss = pid.isna()
    pid[miss] = [uniq.get((sleeper.norm(n), pos)) for n, pos in zip(vol.name[miss], vol.position[miss])]
    return pid


def pregame_rows(d: pd.DataFrame, week: int, vol: pd.DataFrame, dump: dict):
    """Placeholder rows (all stats zero) for the players expected to play in `week`."""
    sched = pd.read_csv(".cache/games.csv", low_memory=False)
    sched = sched[(sched.season == SEASON) & (sched.week == week) & (sched.game_type == "REG")]
    g = pd.concat([sched.assign(team=sched.home_team, opp=sched.away_team), sched.assign(team=sched.away_team, opp=sched.home_team)])[["team", "opp", "gameday"]]
    known = d[d.season == SEASON].sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    v = vol.copy()
    v["player_id"] = map_to_nflverse(v, known, dump)
    out_ids = set(v[v.injury_status.isin(sleeper.OUT_STATUSES)].sleeper_id)
    v = v.merge(g, on="team", how="inner")                              # teams on a bye drop out
    unmapped = v[v.player_id.isna() & ~v.sleeper_id.isin(out_ids)]
    v = v[v.player_id.notna() & ~v.sleeper_id.isin(out_ids)].drop_duplicates("player_id")
    ph = pd.DataFrame(0.0, index=range(len(v)), columns=d.columns)
    ph["player_id"], ph["player_display_name"], ph["position"] = v.player_id.to_numpy(), v.name.to_numpy(), v.position.to_numpy()
    ph["team"], ph["opp"], ph["season"], ph["week"] = v.team.to_numpy(), v.opp.to_numpy(), SEASON, week
    ph["pass_att"] = v.attempts.to_numpy()                              # lets the QB of the game be identified
    ph[["snap_pct", "carry_share"]] = np.nan
    ph["day"] = (pd.to_datetime(v.gameday.to_numpy()) - pd.Timestamp("2020-01-01")).days.astype(float)
    ph["pts"], ph["pts_notd"] = np.nan, np.nan
    return ph, known, unmapped


def pregame_features(week: int, vol: pd.DataFrame, dump: dict):
    d = MD.raw_table()
    ph, known, unmapped = pregame_rows(d, week, vol, dump)
    d2 = pd.concat([d, ph], ignore_index=True).sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    base = ph[["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts", "pts_notd"]]
    f = base.merge(MD._player_features(d2, SEASON), on=["player_id", "season", "week"], how="left")
    f = f.merge(MD._team_context(d2, SEASON).drop(columns=["team", "opp", "position"]), on=["player_id", "season", "week"], how="left")
    f = f.merge(MD._availability(d2, SEASON).drop(columns="team"), on=["player_id", "season", "week"], how="left")
    f = f.merge(MD._quarterback(d2, SEASON), on=["player_id", "season", "week"], how="left")
    form = f[f.player_id == f.qb_id][["season", "week", "team"] + MD.QB_FORM].rename(columns={c: f"qbs_{c}" for c in MD.QB_FORM}).drop_duplicates(["season", "week", "team"])
    f = f.merge(form, on=["season", "week", "team"], how="left").drop(columns="qb_id")
    sched = pd.read_csv(".cache/games.csv", low_memory=False)
    sched = sched[(sched.season == SEASON) & (sched.week == week) & (sched.game_type == "REG")]
    h = sched.assign(team=sched.home_team, home=1.0, fav=sched.spread_line, implied=sched.total_line / 2 + sched.spread_line / 2)
    a = sched.assign(team=sched.away_team, home=0.0, fav=-sched.spread_line, implied=sched.total_line / 2 - sched.spread_line / 2)
    veg = pd.concat([h, a])[["season", "week", "team", "home", "fav", "implied", "total_line"]]
    f = f.merge(veg, on=["season", "week", "team"], how="left")
    sl = MD.sleeper_points(SEASON, week, known, dump)
    f = f.merge(sl, on=["player_id", "season", "week"], how="left")
    f = pd.concat([f, pd.DataFrame({f"is_{p}": (f.position == p).astype(float) for p in MD.POS})], axis=1)
    return f, unmapped


def best_lineup(df: pd.DataFrame, col: str):
    """QB, RB, RB, WR, WR, TE, FLEX, FLEX with the highest total of `col`; returns (total, set of names)."""
    pos = {p: df[df.position == p].sort_values(col, ascending=False) for p in MD.POS}
    best = (-1.0, None)
    qb = pos["QB"].head(1)
    if qb.empty:
        return best
    skill = df[df.position.isin(["RB", "WR", "TE"])].sort_values(col, ascending=False).head(10)
    for combo in itertools.combinations(skill.index, 7):
        c = skill.loc[list(combo)]
        if (c.position == "RB").sum() >= 2 and (c.position == "WR").sum() >= 2 and (c.position == "TE").sum() >= 1:
            tot = c[col].sum() + float(qb[col].iloc[0])
            if tot > best[0]:
                best = (tot, set(c.name_) | set(qb.name_))
    return best


def week_rows(week: int, refresh: bool = False, max_age_h: float = 6.0) -> pd.DataFrame:
    """Pre-game rows for every projected player of the week with the ensemble's and Sleeper's projections (cached for a few hours)."""
    import time
    from pathlib import Path
    cache = Path(f".cache/ml_week_{SEASON}_{week}.pkl")
    if cache.exists() and not refresh and (time.time() - cache.stat().st_mtime) / 3600 < max_age_h:
        return pd.read_pickle(cache)
    dump = sleeper.load_players()
    vol = A.sleeper_volume(SEASON, week)
    F, _ = pregame_features(week, vol, dump)
    hist = pd.read_pickle(MD.OUT)
    hist = hist[hist.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(hist)["avail+qb+cal"]
    te = F[F.n_hist >= 1].copy()
    pr = fit_predict(hist[hist.season.between(2021, 2025)], te, feats)
    for c in pr:
        te[c] = pr[c]
    te["blend"] = np.where(te.sl.notna(), 0.5 * te.sl + 0.5 * te.ensemble, np.nan)
    pw = data.load_player_weeks(SEASON)
    known = pw[pw.position.isin(MD.POS)].sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    te["sleeper_id"] = te.player_id.map(dict(zip(map_to_nflverse(vol, known, dump), vol.sleeper_id)))
    te.to_pickle(cache)
    return te


if __name__ == "__main__":
    pd.options.display.width = 250
    user = sys.argv[1]
    week = int(sys.argv[2]) if len(sys.argv) > 2 else sleeper.get("state/nfl")["week"]
    uid = sleeper.user_id(user)
    lg = sleeper.get(f"user/{uid}/leagues/nfl/{SEASON}")[0]
    scoring, _ = league.scoring_from_league(lg["scoring_settings"])
    dump = sleeper.load_players()
    vol = A.sleeper_volume(SEASON, week)
    F, unmapped = pregame_features(week, vol, dump)
    print(f"week {week}: {len(F)} pre-game rows built (players Sleeper projects who are not Out/IR); {len(unmapped)} projected players could not be matched to nflverse")
    hist = pd.read_pickle(MD.OUT)
    hist = hist[hist.n_hist >= 1].reset_index(drop=True)
    feats = feature_sets(hist)["avail+qb+cal"]
    miss = [c for c in feats if c not in F.columns]
    assert not miss, f"missing inputs: {miss}"
    print("inputs:", len(feats), "| rows with at least one earlier game since last season:", int((F.n_hist >= 1).sum()), "of", len(F))
    tr = hist[hist.season.between(2021, 2025)]
    te = F[F.n_hist >= 1].copy()
    pr = fit_predict(tr, te, feats)
    for c in pr:
        te[c] = pr[c]
    te["blend"] = np.where(te.sl.notna(), 0.5 * te.sl + 0.5 * te.ensemble, np.nan)
    trs = tr[tr.sl.notna()]
    sf = feats + SL_FEATS
    ok = te.sl.notna()
    te.loc[ok, "stacked"] = np.mean([gbm("squared_error", s).fit(trs[sf], trs.pts).predict(te.loc[ok, sf]) for s in range(3)], axis=0).clip(min=0)
    te.to_pickle("/tmp/ml_week_rows.pkl")

    # the hand-built modules 1-4
    from analysis.week_players import project
    hb, _ = project(SEASON, week, scoring, active_only=False)
    hb2, _ = project(SEASON, week, scoring, active_only=True)
    hb_map = dict(zip(hb.sleeper_id, hb.total_pts))
    hb2_map = dict(zip(hb2.sleeper_id, hb2.total_pts))
    vm = vol.assign(player_id=map_to_nflverse(vol, MD.raw_table().query("season == @SEASON").sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index(), dump))
    sid_of = dict(zip(vm.player_id, vm.sleeper_id))
    te["sleeper_id"] = te.player_id.map(sid_of)
    te["modules"] = te.sleeper_id.map(hb_map)
    te["modules_active"] = te.sleeper_id.map(hb2_map)
    te.to_pickle("/tmp/ml_week_rows.pkl")

    # roster
    rosters = sleeper.get(f"league/{lg['league_id']}/rosters")
    me = next(r for r in rosters if r["owner_id"] == uid)
    mu = sleeper.get(f"league/{lg['league_id']}/matchups/{week}")
    mine = next((x for x in mu if x["roster_id"] == me["roster_id"]), None)
    starters = set(mine["starters"]) if mine else set(me["starters"])
    rows, none = [], []
    for sid in me["players"]:
        p = dump.get(sid, {})
        if p.get("position") not in MD.POS:
            continue
        name = f'{p.get("first_name")} {p.get("last_name")}'
        r = te[te.sleeper_id == sid]
        if r.empty:
            none.append(f'{name} ({p.get("position")} {p.get("team")}, {p.get("injury_status") or "no status"}{", taxi" if sid in (me.get("taxi") or []) else ""})')
            continue
        r = r.iloc[0]
        rows.append(dict(name_=name, position=r.position, team=r.team, opp=r.opp, start=("S" if sid in starters else ""), status=p.get("injury_status") or "",
                         sleeper=r.sl, modules=r.modules, modules_active=r.modules_active, ridge=r.ridge, trees=r.trees, trees_poisson=r.trees_poisson,
                         neural_net=r.neural_net, ensemble=r.ensemble, blend=r.blend, stacked=r.stacked))
    R = pd.DataFrame(rows)
    cols = ["sleeper", "modules", "modules_active", "ridge", "trees", "trees_poisson", "neural_net", "ensemble", "blend", "stacked"]
    R["avg_all"] = R[cols].mean(axis=1)
    R["spread"] = R[cols].max(axis=1) - R[cols].min(axis=1)
    R = R.sort_values("avg_all", ascending=False).reset_index(drop=True)
    R.to_pickle("/tmp/ml_week_roster.pkl")
    print(f"\n{lg['name']}, week {week}, scoring {scoring}\n")
    print(R.rename(columns={"name_": "player", "start": "S", "trees_poisson": "trees_poi", "neural_net": "nn", "modules_active": "mod_act"}).to_string(float_format="{:.1f}".format))
    print("\nno projection:", "; ".join(none) or "none")
    print("\nbest lineup by each model (QB, 2 RB, 2 WR, TE, 2 FLEX), sum of that model's points:")
    for c in cols + ["avg_all"]:
        tot, names = best_lineup(R.rename(columns={}).assign(name_=R.name_), c)
        print(f"  {c:15s} {tot:6.1f} | " + ", ".join(sorted(names)))
    print("\ncurrent starters:", ", ".join(sorted(R[R.start == "S"].name_)))
