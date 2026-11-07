"""Same-season predictability of touchdowns: how well does opportunity (carries / targets, and distance-aware expected TDs) track actual TDs?

Expected TD per play = the league's scoring rate for that play type at that yard line (1-yard bins inside the 20, 5-yard bins beyond), fitted on the same seasons.
    uv run python -m analysis.td_same_year
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["season", "week", "season_type", "posteam", "defteam", "play_type", "two_point_attempt", "qb_kneel", "qb_spike", "yardline_100",
        "rush_attempt", "pass_attempt", "rush_touchdown", "pass_touchdown", "rusher_player_id", "receiver_player_id", "passer_player_id",
        "rushing_yards", "receiving_yards", "complete_pass", "sack"]


def load(seasons=(2024, 2025)) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f".cache/pbp_{y}.csv.gz", usecols=COLS, low_memory=False) for y in seasons], ignore_index=True)
    d = d[(d.season_type == "REG") & d.posteam.notna() & d.play_type.isin(["run", "pass"]) & (d.two_point_attempt != 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)]
    d["bin"] = np.where(d.yardline_100 <= 20, d.yardline_100, 20 + 5 * np.ceil((d.yardline_100 - 20) / 5)).astype(float)
    return d


def add_xtd(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["td"] = np.where(d.play_type == "run", d.rush_touchdown, d.pass_touchdown).astype(float)
    d["xtd"] = d.groupby(["play_type", "bin"]).td.transform("mean")
    return d


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    d = add_xtd(load())
    rush = d[(d.play_type == "run") & d.rusher_player_id.notna()].assign(pid=lambda x: x.rusher_player_id)
    rec = d[(d.play_type == "pass") & d.receiver_player_id.notna()].assign(pid=lambda x: x.receiver_player_id)
    for label, p, minn in (("RUSHING TDs (rusher)", rush, 40), ("RECEIVING TDs (target)", rec, 30)):
        print(f"\n===== {label} =====")
        for scope in ("season", "game"):
            keys = ["season", "pid"] if scope == "season" else ["season", "week", "pid"]
            g = p.groupby(keys).agg(opp=("td", "size"), td=("td", "sum"), xtd=("xtd", "sum"), team=("posteam", "last")).reset_index()
            yd = p.groupby(keys)[("rushing_yards" if "RUSH" in label else "receiving_yards")].sum().rename("yds").reset_index()
            g = g.merge(yd, on=keys)
            if scope == "season":
                g = g[g.opp >= minn]
            print(f"  {scope}-level, n={len(g)}: corr(TDs, opportunities) {corr(g.td, g.opp):.3f} | corr(TDs, yards) {corr(g.td, g.yds):.3f} | "
                  f"corr(TDs, xTD) {corr(g.td, g.xtd):.3f} (R2 {corr(g.td, g.xtd)**2:.3f})")
        # split-half reliability of TDs within a season: odd weeks vs even weeks, same players
        h = p.assign(half=p.week % 2).groupby(["season", "pid", "half"]).agg(opp=("td", "size"), td=("td", "sum"), xtd=("xtd", "sum")).unstack("half").dropna()
        h = h[(h[("opp", 0)] >= minn / 2) & (h[("opp", 1)] >= minn / 2)]
        print(f"  odd-vs-even weeks, n={len(h)} players: corr(TDs, TDs) {corr(h[('td', 0)], h[('td', 1)]):.3f} | corr(xTD, xTD) {corr(h[('xtd', 0)], h[('xtd', 1)]):.3f} | "
              f"corr(xTD in one half, TDs in the other) {np.mean([corr(h[('xtd', 0)], h[('td', 1)]), corr(h[('xtd', 1)], h[('td', 0)])]):.3f} | "
              f"corr(opps in one half, TDs in the other) {np.mean([corr(h[('opp', 0)], h[('td', 1)]), corr(h[('opp', 1)], h[('td', 0)])]):.3f}")
    print("\n===== TEAM-GAME (all teams) =====")
    # team-game: yards from the rush / pass play rows themselves
    d["yds"] = np.where(d.play_type == "run", d.rushing_yards, d.receiving_yards).astype(float)
    for label, pt in (("rush", "run"), ("pass", "pass")):
        t = d[d.play_type == pt].groupby(["season", "week", "posteam"]).agg(td=("td", "sum"), xtd=("xtd", "sum"), plays=("td", "size"), yds=("yds", "sum")).reset_index()
        print(f"  {label}: n={len(t)} team-games, mean TDs {t.td.mean():.2f}: corr(TDs, plays) {corr(t.td, t.plays):.3f} | corr(TDs, yards) {corr(t.td, t.yds):.3f} | corr(TDs, xTD) {corr(t.td, t.xtd):.3f}")
