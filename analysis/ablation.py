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

With --stacked the baseline is instead the ridge model (30% last season / 70% league, K pseudo-games), and the same ideas are added
one at a time on top of it; averages an idea brings with it (the opponent's defense, plays, yards per play) are ridged the same way.

    uv run python -m analysis.ablation              # ideas added to the team average
    uv run python -m analysis.ablation --stacked    # ideas added to the ridge model
"""
from __future__ import annotations

import sys

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


def predict(G: pd.DataFrame, seasons=(2024, 2025, 2026), stacked: bool = False) -> pd.DataFrame:
    teams = sorted(set(G.off_team))
    hist, out = [], []
    cfg = Y.YardsConfig()
    for season in seasons:
        prev = G[G.season == season - 1]
        prev_lg = {s: float(prev[f"{s}_yds"].mean()) for s in SIDES} if len(prev) else None
        prev_avg = prev.groupby("off_team")[[f"{s}_yds" for s in SIDES]].mean() if len(prev) else None
        prev_ypp = {s: prev.groupby("off_team")[f"{s}_yds_pbp"].sum() / prev.groupby("off_team")[f"{s}_att"].sum() for s in SIDES} if len(prev) else None
        prev_att = {s: prev.groupby("off_team")[f"{s}_att"].mean() for s in SIDES} if len(prev) else None
        prior = Y.build_prior(prev[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, cfg) if len(prev) else None
        for week in sorted(G.loc[G.season == season, "week"].unique()):
            cur, todo = G[(G.season == season) & (G.week < week)], G[(G.season == season) & (G.week == week)].copy()
            lg = {s: (float(cur[f"{s}_yds"].mean()) if len(cur) else (prev_lg[s] if prev_lg else np.nan)) for s in SIDES}
            if any(np.isnan(v) for v in lg.values()):
                continue
            n_own = cur.groupby("off_team").size()
            train = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
            train = train[train.season.isin([season - 1, season])] if len(train) else train
            full = Y.fit_ratings(cur[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, cfg, prior, ridge=True, fallback_mu=prev_lg) if stacked else \
                Y.fit_ratings(cur[["off_team", "def_team", "rush_yds", "pass_yds"]], teams, cfg, fallback_mu=prev_lg)
            ref = cur if len(cur) else prev
            l_att = {s: float(ref[f"{s}_att"].mean()) for s in SIDES}
            l_ypp = {s: float(ref[f"{s}_yds_pbp"].sum() / ref[f"{s}_att"].sum()) for s in SIDES}
            for s in SIDES:
                y = f"{s}_yds"
                own = cur.groupby("off_team")[y].mean()
                n = todo.off_team.map(n_own).fillna(0.0)
                pa = todo.off_team.map(prev_avg[y]) if prev_avg is not None else pd.Series(np.nan, index=todo.index)
                team_avg = todo.off_team.map(own).fillna(pa).fillna(lg[s])
                own_or_lg = todo.off_team.map(own).fillna(lg[s])
                prior_mean = 0.3 * pa.fillna(lg[s]) + 0.7 * lg[s]
                ridge_prior = (n * own_or_lg + K * prior_mean) / (n + K)
                base = ridge_prior if stacked else team_avg
                todo[f"{s}_league"] = lg[s]
                todo[f"{s}_base"] = base
                todo[f"{s}_ridge_league"] = (n * own_or_lg + K * lg[s]) / (n + K)
                todo[f"{s}_ridge_prior"] = ridge_prior
                allowed = cur.groupby("def_team")[y].mean()
                n_def = todo.def_team.map(cur.groupby("def_team").size()).fillna(0.0)
                allowed_x = todo.def_team.map(allowed).fillna(lg[s])
                if stacked:
                    allowed_x = (n_def * allowed_x + K * lg[s]) / (n_def + K)       # the opponent's defense gets the same ridge
                todo[f"{s}_opp_onestep"] = base * allowed_x / lg[s]
                todo[f"{s}_opp_iterative"] = [full.expected(o, d)[s] for o, d in zip(todo.off_team, todo.def_team)]
                att_m = cur.groupby("off_team")[f"{s}_att"].mean()
                att_s = cur.groupby("off_team")[f"{s}_att"].sum()
                yds_s = cur.groupby("off_team")[f"{s}_yds_pbp"].sum()
                pat = todo.off_team.map(prev_att[s]) if prev_att else pd.Series(np.nan, index=todo.index)
                ppp = todo.off_team.map(prev_ypp[s]) if prev_ypp else pd.Series(np.nan, index=todo.index)
                if stacked:     # ridge the plays and the yards per play separately, then multiply
                    att_x = (n * todo.off_team.map(att_m).fillna(l_att[s]) + K * (0.3 * pat.fillna(l_att[s]) + 0.7 * l_att[s])) / (n + K)
                    ypp_x = (todo.off_team.map(yds_s).fillna(0.0) + K * l_att[s] * (0.3 * ppp.fillna(l_ypp[s]) + 0.7 * l_ypp[s])) / (todo.off_team.map(att_s).fillna(0.0) + K * l_att[s])
                    todo[f"{s}_pbp"] = att_x * ypp_x
                else:
                    a_ = todo.off_team.map(att_m).fillna(pat)
                    p_ = (todo.off_team.map(yds_s) / todo.off_team.map(att_s)).fillna(ppp)
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
                b = todo[f"{s}_base"].to_numpy(float)
                sign = 1.0 if s == "rush" else -1.0
                todo[f"{s}_vegas_free"] = np.maximum(b + slope[s] * todo.fav.to_numpy(float), 0.0)
                todo[f"{s}_vegas_transfer"] = np.maximum(b + sign * beta * todo.fav.to_numpy(float), 0.0)
                todo[f"{s}_wind"] = np.maximum(b + wsl[s] * xt, 0.0)
            todo["slope_rush"], todo["slope_pass"], todo["beta"], todo["wind_rush"], todo["wind_pass"] = slope["rush"], slope["pass"], beta, wsl["rush"], wsl["pass"]
            out.append(todo)
            hist.append(todo[["season", "week", "fav", "wind5", "rush_yds", "pass_yds", "rush_base", "pass_base"]])
    return pd.concat(out, ignore_index=True)


def delta_vs_baseline(bt: pd.DataFrame, side: str, col: str, n_boot: int = 3000) -> tuple[float, float, float]:
    """Change in R2 against the baseline with a 95% interval from resampling team-games (positive = better)."""
    a, p, b = bt[f"{side}_yds"].to_numpy(float), bt[col].to_numpy(float), bt[f"{side}_base"].to_numpy(float)
    d = (a - b) ** 2 - (a - p) ** 2
    var = ((a - a.mean()) ** 2).mean()
    idx = np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1) / var
    return d.mean() / var, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def table(bt: pd.DataFrame, side: str, stacked: bool = False) -> pd.DataFrame:
    rows = {}
    variants = ["league", "base"] + ([v for v in VARIANTS if v not in ("team_avg", "ridge_league", "ridge_prior")] if stacked else [v for v in VARIANTS if v != "team_avg"])
    for v in variants:
        col = f"{side}_{v}"
        m = metrics(bt, side, col)
        d = delta_vs_baseline(bt, side, col) if v not in ("league", "base") else (np.nan, np.nan, np.nan)
        name = ("BASELINE: ridge (30% last season / 70% league)" if stacked else LABEL["team_avg"]) if v == "base" else LABEL[v]
        rows[name] = dict(R2=m["R2"], MAE=m["MAE"], bias=m["bias"], corr=m["corr"], pairs=m["pairs"], dR2=d[0], lo=d[1], hi=d[2])
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 230
    stacked = "--stacked" in sys.argv
    G = load_all()
    P = predict(G, stacked=stacked)
    bt = P[P.season.isin([2025, 2026])]
    print(f"{len(bt[bt.week >= MAIN_FROM])} team-games, weeks {MAIN_FROM}+ of 2025-26 (ridge strength {K:.0f}; baseline: {'the ridge model' if stacked else 'the team average'}); dR2 = change in R2 vs the baseline, with a 95% interval\n")
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), (f"weeks 1-{MAIN_FROM - 1}", bt[bt.week < MAIN_FROM])):
        for side in SIDES:
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games; actual mean {sub[f'{side}_yds'].mean():.1f}, sd {sub[f'{side}_yds'].std():.1f})")
            print(table(sub, side, stacked).to_string()); print()
    last = bt[bt.week >= MAIN_FROM].groupby(["season", "week"])[["slope_rush", "slope_pass", "beta", "wind_rush", "wind_pass"]].first()
    print("fitted effects at the end of each season (yards per point favoured: rush, pass | transfer beta | yards per 5 mph of wind: rush, pass):")
    print(last.groupby("season").tail(1).round(2).to_string())
