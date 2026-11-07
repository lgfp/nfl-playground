"""Player-level backtest, 2026 weeks 4-8: actual fantasy points vs Sleeper's projections vs ours (modules 1-4, each week from earlier games only).

    uv run python -m analysis.backtest_players [first_week] [last_week]
Caveat: Sleeper serves past weeks' projections only as they stand today (stamped the Tuesday after the games), and the injury status in the feed is today's, so
neither side excludes injured players; our module 2 uses the same Sleeper volume for that week. Wind in module 1 is the game's actual wind.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import attribution as A, data, sleeper, td_attribution as TA, team_tds as M3, team_yards as TY, tds as T
from nflfp.model import Scoring

SEASON = 2026


def actual_points(pw: pd.DataFrame, sc: Scoring) -> pd.DataFrame:
    a = pw.copy()
    for c in ("rushing_tds", "receiving_tds", "passing_tds"):
        a[c] = a[c].fillna(0)
    a["act_notd"] = sc.yard * (a.rushing_yards + a.receiving_yards) + sc.ppr * a.receptions + sc.pass_yard * a.passing_yards
    a["act"] = a.act_notd + sc.td * (a.rushing_tds + a.receiving_tds) + sc.pass_td * a.passing_tds
    a["act_rush_yds"], a["act_rec_yds"] = a.rushing_yards, a.receiving_yards
    return a[["player_id", "week", "act", "act_notd", "act_rush_yds", "act_rec_yds", "passing_yards"]]


def run(first: int, last: int, active_only: bool = False) -> pd.DataFrame:
    sc = Scoring()
    games, _ = load()
    env = load_env()
    yards = TY.walk_forward(games, env, [SEASON - 1, SEASON])
    plays = T.with_xtd(T.load_plays([SEASON - 1, SEASON]))
    imp = pd.concat([M3.implied_totals(data.load_schedule(y)) for y in (SEASON - 1, SEASON)], ignore_index=True)
    ttd = M3.walk_forward(T.team_game_tds(plays), yards, imp, [SEASON])
    pw = data.load_player_weeks(SEASON)
    known = pw.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    dump = sleeper.load_players()
    act = actual_points(pw, sc)
    parts = []
    for w in range(first, last + 1):
        vol = A.sleeper_volume(SEASON, w)
        teams = yards[(yards.season == SEASON) & (yards.week == w)]
        m2 = A.attribute(teams, A.shares(vol, out_statuses=()), A.league_rates(pw[pw.week < w]), sc)
        excl = set()
        if active_only:        # TD shares only among players Sleeper projects with volume that week (the rest are out / inactive)
            have = set(sleeper.map_ids(dump, known, vol.sleeper_id).values())
            excl = set(TA.shares(plays, SEASON, w).pid) - have
        td = TA.attribute(ttd[(ttd.season == SEASON) & (ttd.week == w)], TA.shares(plays, SEASON, w, exclude=excl))
        full = TA.add_touchdowns(m2, td, sleeper.map_ids(dump, known, m2.sleeper_id), sc)
        full["pid"] = full.sleeper_id.map(sleeper.map_ids(dump, known, full.sleeper_id))
        full["week"] = w
        full["sl_notd"] = sc.yard * (full.sl_rush_yds + full.sl_rec_yds) + sc.ppr * full.sl_receptions + sc.pass_yard * full.sl_pass_yds
        full["sl"] = full.sl_notd + sc.td * (full.sl_rush_tds + full.sl_rec_tds) + sc.pass_td * full.sl_pass_tds
        parts.append(full)
    P = pd.concat(parts, ignore_index=True)
    P = P.merge(act.rename(columns={"player_id": "pid"}), on=["pid", "week"], how="left")
    P["played"] = P.act.notna()
    P[["act", "act_notd"]] = P[["act", "act_notd"]].fillna(0.0)
    P["ours"], P["ours_notd"] = P.total_pts, P.exp_pts
    return P


def table(d: pd.DataFrame, a: str, label: str):
    rows = []
    for name, col in (("Sleeper", "sl" if a == "act" else "sl_notd"), ("Ours (modules 1-4)" if a == "act" else "Ours (modules 1-2)", "ours" if a == "act" else "ours_notd")):
        e = d[col] - d[a]
        rows.append(dict(source=name, MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), corr=np.corrcoef(d[col], d[a])[0, 1], R2=1 - (e ** 2).sum() / ((d[a] - d[a].mean()) ** 2).sum()))
    e = (d["sl" if a == "act" else "sl_notd"] + d["ours" if a == "act" else "ours_notd"]) / 2 - d[a]
    rows.append(dict(source="Average of both", MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), corr=np.corrcoef((d["sl" if a == "act" else "sl_notd"] + d["ours" if a == "act" else "ours_notd"]) / 2, d[a])[0, 1],
                     R2=1 - (e ** 2).sum() / ((d[a] - d[a].mean()) ** 2).sum()))
    print(f"\n{label}: n={len(d)}, mean actual {d[a].mean():.2f}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 200
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    P = run(first, last)
    P.to_pickle("/tmp/backtest_players.pkl")
    pool = P[(P.sl >= 4) & P.pid.notna()]
    print(f"weeks {first}-{last}, 2026. pool = players Sleeper projects at 4+ points (with a match to nflverse): {len(pool)} player-games; "
          f"{(~pool.played).sum()} of them did not play ({pool[~pool.played].sl.mean():.1f} Sleeper pts, {pool[~pool.played].ours.mean():.1f} ours on average)")
    table(pool, "act", "ALL projected (incl. players who did not play), full points")
    pl = pool[pool.played]
    table(pl, "act", "PLAYED only, full points (yards + receptions + TDs)")
    table(pl, "act_notd", "PLAYED only, without touchdowns")
    for pos in ("QB", "RB", "WR", "TE"):
        table(pl[pl.position == pos], "act", f"PLAYED only, {pos}, full points")
    for w, d in pl.groupby("week"):
        table(d, "act", f"PLAYED only, week {w}")
