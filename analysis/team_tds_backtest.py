"""Backtest of modules 3 and 4 as shipped (nflfp/team_tds.py, nflfp/td_attribution.py), walk-forward on 2025-26 weeks 4+.

    uv run python -m analysis.team_tds_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import data, td_attribution as TA, team_tds as M3, team_yards as TY, tds as T

MAIN_FROM = 4


def skill(y, p, base):
    return 1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum()


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    plays = T.with_xtd(T.load_plays([2024, 2025, 2026]))
    tg = T.team_game_tds(plays)
    games, _ = load()
    yards = TY.walk_forward(games, load_env(), [2024, 2025, 2026])
    imp = pd.concat([M3.implied_totals(data.load_schedule(y)) for y in (2024, 2025, 2026)], ignore_index=True)
    out = M3.walk_forward(tg, yards, imp, [2024, 2025, 2026])
    # league average to date as the baseline
    lg = []
    for (s, w), g in out.groupby(["season", "week"]):
        past = tg[((tg.season == s) & (tg.week < w)) | (tg.season == s - 1)]
        lg.append(g[M3.KEYS].assign(lg_rush=past.rush_td.mean(), lg_pass=past.pass_td.mean()))
    out = out.merge(pd.concat(lg), on=M3.KEYS)
    ev = out[(out.season >= 2025) & (out.week >= MAIN_FROM) & out.rush_td.notna()]
    print(f"MODULE 3 - team TDs per team-game, 2025-26 weeks {MAIN_FROM}+, n={len(ev)}")
    for side in ("rush", "pass"):
        y, p, b = ev[f"{side}_td"].to_numpy(), ev[f"{side}_td_exp"].to_numpy(), ev[f"lg_{side}"].to_numpy()
        print(f"  {side}: mean actual {y.mean():.3f}, model mean {p.mean():.3f}, MAE {np.abs(y - p).mean():.3f} (league avg {np.abs(y - b).mean():.3f}), "
              f"corr {np.corrcoef(p, y)[0, 1]:.3f}, MSE skill vs league average {skill(y, p, b):.3f}")
    y, p, b = (ev.rush_td + ev.pass_td).to_numpy(), ev.total_td_exp.to_numpy(), (ev.lg_rush + ev.lg_pass).to_numpy()
    print(f"  total: corr {np.corrcoef(p, y)[0, 1]:.3f}, MSE skill {skill(y, p, b):.3f}")

    # module 4: per-player TDs; candidates = players with usage in the team's last 6 games (8 carries / 12 targets), as in analysis/td_player.py
    print("\nMODULE 4 - player TDs per game vs the mean candidate rate")
    for kind, min_recent, col in (("rush", 8, "rush_tds"), ("rec", 12, "rec_tds")):
        play_type, pid = ("run", "rusher_player_id") if kind == "rush" else ("pass", "receiver_player_id")
        pl = plays[(plays.play_type == play_type) & plays[pid].notna()]
        rows = []
        for (s, w), g in ev.groupby(["season", "week"]):
            td = TA.attribute(g, TA.shares(plays, s, w))
            hist = pl[(pl.season >= s - 1) & ((pl.season < s) | (pl.week < w))]
            cand = []
            for team, gg in hist.groupby("posteam"):
                last6 = gg[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(6)
                n = gg.merge(last6, on=["season", "week"]).groupby(pid).size()
                cand.append(pd.DataFrame({"team": team, "pid": n[n >= min_recent].index}))
            act = pl[(pl.season == s) & (pl.week == w)].groupby(["posteam", pid]).td.sum().rename("td").reset_index().rename(columns={"posteam": "team", pid: "pid"})
            rows.append(pd.concat(cand).merge(td[["team", "pid", col]], on=["team", "pid"], how="left").merge(act, on=["team", "pid"], how="left").fillna({col: 0.0, "td": 0.0}))
        D = pd.concat(rows)
        base = D.td.mean()
        print(f"  {kind}: n={len(D)} candidate-games, mean TD/game {base:.3f}, corr {np.corrcoef(D[col], D.td)[0, 1]:.3f}, "
              f"MSE skill vs the mean candidate {skill(D.td.to_numpy(), D[col].to_numpy(), np.full(len(D), base)):.3f}")
