"""One idea at a time, on top of the simplest model: a team's own average yards to date.

Baseline: each team's average rush / pass yards in the current season's earlier games (last season's average in week 1).
Each variant below changes exactly ONE thing about the baseline - nothing is stacked:

  ridge (league)   shrink the team average toward the current league average (K pseudo-games)
  ridge (30/70)    shrink toward 30% last season's team average + 70% league average
  opp (one step)   scale the baseline by how many yards the next opponent's defense allowed per game vs the league
  opp (iterative)  the full iterative opponent adjustment (module 1, no ridge): MU * OFF * DEF
  Vegas (free)     baseline + (yards per point favoured, fitted on earlier games) * spread, rush and pass separately
  Vegas (transfer) baseline +/- one yard-for-yard transfer between rush and pass
  play-by-play     team's average plays x its pooled yards per play (no adjustment of any kind)
  wind             baseline + (yards per 5 mph, fitted on earlier games) * (wind - average), rush and pass separately

Walk-forward on 2025 and 2026 (season 2025's "last season" is 2024); every fit uses only earlier games.

    uv run python -m analysis.ablation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.plays_backtest import KEYS, load_plays
from analysis.yards_backtest import MAIN_FROM, load, metrics
from nflfp import data, lines as linemod, weather, yards as Y

SIDES = ("rush", "pass")
K = 8.0
VARIANTS = ["team_avg", "ridge_league", "ridge_prior", "opp_onestep", "opp_iterative", "vegas_free", "vegas_transfer", "pbp", "wind"]
LABEL = {"league": "league average to date", "team_avg": "BASELINE: team average", "ridge_league": "+ ridge toward league average", "ridge_prior": "+ ridge toward 30% last season / 70% league",
         "opp_onestep": "+ opponent's raw defense (one step)", "opp_iterative": "+ opponent adjustment (iterative)", "vegas_free": "+ Vegas lines (free slopes)",
         "vegas_transfer": "+ Vegas lines (rush<->pass transfer)", "pbp": "+ play-by-play (plays x yds/play)", "wind": "+ wind"}


def load_all() -> pd.DataFrame:
    games, fixtures = load()
    gp = load_plays().rename(columns={"rush_yds": "rush_yds_pbp", "pass_yds": "pass_yds_pbp"})
    wind = []
    for y in (2024, 2025, 2026):
        sched = data.load_schedule(y)
        tl = linemod.team_lines(sched, weather.game_weather(sched))
        wind.append(tl.assign(season=y).rename(columns={"team": "off_team"})[["season", "week", "off_team", "wind5"]])
    G = games.merge(fixtures, on=KEYS).merge(pd.concat(wind), on=["season", "week", "off_team"], how="left").merge(gp, on=KEYS)
    return G.sort_values(["season", "week"]).reset_index(drop=True)


def predict(G: pd.DataFrame, seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    teams = sorted(set(G.off_team))
    hist, out = [], []
    cfg = Y.YardsConfig()
    for season in seasons:
        prev = G[G.season == season - 1]
        prev_lg = {s: float(prev[f"{s}_yds"].mean()) for s in SIDES} if len(prev) else None
        prev_avg = prev.groupby("off_team")[[f"{s}_yds" for s in SIDES]].mean() if len(prev) else None
        prev_ypp = {s: prev.groupby("off_team")[f"{s}_yds_pbp"].sum() / prev.groupby("off_team")[f"{s}_att"].sum() for s in SIDES} if len(prev) else None
        prev_att = {s: prev.groupby("off_team")[f"{s}_att"].mean() for s in SIDES} if len(prev) else None
        for week in sorted(G.loc[G.season == season, "week"].unique()):
            cur, todo = G[(G.season == season) & (G.week < week)], G[(G.season == season) & (G.week == week)].copy()
            lg = {s: (float(cur[f"{s}_yds"].mean()) if len(cur) else (prev_lg[s] if prev_lg else np.nan)) for s in SIDES}
            if any(np.isnan(v) for v in lg.values()):
                continue
            n_own = cur.groupby("off_team").size()
            train = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
            train = train[train.season.isin([season - 1, season])] if len(train) else train
            plain = Y.fit_ratings(cur[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, cfg, fallback_mu=prev_lg) if (len(cur) or prev_lg) else None
            for s in SIDES:
                y = f"{s}_yds"
                own = cur.groupby("off_team")[y].mean()
                n = todo.off_team.map(n_own).fillna(0.0)
                pa = todo.off_team.map(prev_avg[y]) if prev_avg is not None else pd.Series(np.nan, index=todo.index)
                base = todo.off_team.map(own).fillna(pa).fillna(lg[s])
                todo[f"{s}_league"] = lg[s]
                todo[f"{s}_team_avg"] = base
                own_or_lg = todo.off_team.map(own).fillna(lg[s])
                todo[f"{s}_ridge_league"] = (n * own_or_lg + K * lg[s]) / (n + K)
                prior_mean = 0.3 * pa.fillna(lg[s]) + 0.7 * lg[s]
                todo[f"{s}_ridge_prior"] = (n * own_or_lg + K * prior_mean) / (n + K)
                allowed = cur.groupby("def_team")[y].mean()
                todo[f"{s}_opp_onestep"] = base * todo.def_team.map(allowed).fillna(lg[s]) / lg[s]
                todo[f"{s}_opp_iterative"] = [plain.expected(o, d)[s] for o, d in zip(todo.off_team, todo.def_team)]
                att = cur.groupby("off_team")[f"{s}_att"].mean()
                ypp = cur.groupby("off_team")[f"{s}_yds_pbp"].sum() / cur.groupby("off_team")[f"{s}_att"].sum()
                a_ = todo.off_team.map(att).fillna(todo.off_team.map(prev_att[s]) if prev_att else np.nan)
                p_ = todo.off_team.map(ypp).fillna(todo.off_team.map(prev_ypp[s]) if prev_ypp else np.nan)
                todo[f"{s}_pbp"] = (a_ * p_).fillna(lg[s])
            # lines and wind: slopes fitted on the baseline's residuals in last season + this season's earlier weeks
            f_tr = train.fav.to_numpy(float) if len(train) else np.array([])
            res = {s: (train[f"{s}_yds"] - train[f"{s}_base"]).to_numpy(float) for s in SIDES} if len(train) else {}
            ss = float((f_tr ** 2).sum()) if len(train) else 0.0
            slope = {s: (float((f_tr * res[s]).sum() / ss) if ss else 0.0) for s in SIDES}
            beta = float((f_tr * (res["rush"] - res["pass"])).sum() / (2 * ss)) if ss else 0.0
            ok = train.wind5.notna().to_numpy() if len(train) else np.array([], bool)
            wmean = float(train.wind5[ok].mean()) if ok.any() else 0.0
            xw = (train.wind5.to_numpy(float)[ok] - wmean) if ok.any() else np.array([])
            wsl = {s: (float((xw * res[s][ok]).sum() / (xw ** 2).sum()) if len(xw) and (xw ** 2).sum() else 0.0) for s in SIDES}
            xt = (todo.wind5.fillna(wmean) - wmean).to_numpy(float)
            for s in SIDES:
                b = todo[f"{s}_team_avg"].to_numpy(float)
                sign = 1.0 if s == "rush" else -1.0
                todo[f"{s}_vegas_free"] = np.maximum(b + slope[s] * todo.fav.to_numpy(float), 0.0)
                todo[f"{s}_vegas_transfer"] = np.maximum(b + sign * beta * todo.fav.to_numpy(float), 0.0)
                todo[f"{s}_wind"] = np.maximum(b + wsl[s] * xt, 0.0)
            todo["slope_rush"], todo["slope_pass"], todo["beta"], todo["wind_rush"], todo["wind_pass"] = slope["rush"], slope["pass"], beta, wsl["rush"], wsl["pass"]
            out.append(todo)
            hist.append(todo[["season", "week", "fav", "wind5", "rush_yds", "pass_yds"]].assign(rush_base=todo.rush_team_avg, pass_base=todo.pass_team_avg))
    return pd.concat(out, ignore_index=True)


def delta_vs_baseline(bt: pd.DataFrame, side: str, col: str, n_boot: int = 3000) -> tuple[float, float, float]:
    """Change in R2 against the baseline with a 95% interval from resampling team-games (positive = better)."""
    a, p, b = bt[f"{side}_yds"].to_numpy(float), bt[col].to_numpy(float), bt[f"{side}_team_avg"].to_numpy(float)
    d = (a - b) ** 2 - (a - p) ** 2
    var = ((a - a.mean()) ** 2).mean()
    idx = np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1) / var
    return d.mean() / var, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def table(bt: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = {}
    for v in ["league"] + VARIANTS:
        col = f"{side}_{v}"
        m = metrics(bt, side, col)
        d = delta_vs_baseline(bt, side, col) if v not in ("league", "team_avg") else (np.nan, np.nan, np.nan)
        rows[LABEL[v]] = dict(R2=m["R2"], MAE=m["MAE"], bias=m["bias"], corr=m["corr"], pairs=m["pairs"], dR2=d[0], lo=d[1], hi=d[2])
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 230
    G = load_all()
    P = predict(G)
    bt = P[P.season.isin([2025, 2026])]
    print(f"{len(bt[bt.week >= MAIN_FROM])} team-games, weeks {MAIN_FROM}+ of 2025-26 (ridge strength {K:.0f}); dR2 = change in R2 vs the baseline, with a 95% interval\n")
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), (f"weeks 1-{MAIN_FROM - 1}", bt[bt.week < MAIN_FROM])):
        for side in SIDES:
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games; actual mean {sub[f'{side}_yds'].mean():.1f}, sd {sub[f'{side}_yds'].std():.1f})")
            print(table(sub, side).to_string()); print()
    last = bt[bt.week >= MAIN_FROM].groupby(["season", "week"])[["slope_rush", "slope_pass", "beta", "wind_rush", "wind_pass"]].first()
    print("fitted effects at the end of each season (yards per point favoured: rush, pass | transfer beta | yards per 5 mph of wind: rush, pass):")
    print(last.groupby("season").tail(1).round(2).to_string())
