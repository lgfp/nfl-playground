"""Stacking ideas on the ridge model: rush = ridge + Vegas (free slopes), pass = ridge + iterative opponent adjustment.

A pipeline is a base estimate followed by additive corrections applied in order (Vegas: yards per point favoured; wind: yards per 5 mph
around the average). Every slope is fitted on the residuals of the stage before it, on last season plus this season's earlier weeks.

Round 2 (this file): the new base for each side, then each remaining idea added ALONE on top of it:
    rush:  base = ridge -> Vegas        + opponent adjustment | + wind | + play-by-play
    pass:  base = ridge + opponent adj. + Vegas | + wind | + play-by-play (module 1b)
Walk-forward on 2025-2026 (season 2025's "last season" is 2024).

    uv run python -m analysis.stack            # round 2
    uv run python -m analysis.stack --round3   # passing only: wind joins the pass base (ridge + opponent + wind); then + Vegas | + play-by-play
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.ablation import K, load_all
from analysis.yards_backtest import MAIN_FROM, metrics
from nflfp import plays as P, yards as Y

SIDES = ("rush", "pass")
# name -> (base estimator, corrections)
PIPES = {
    "rush": {"rush_ridge": ("ridge", []), "rush_BASE (ridge+Vegas)": ("ridge", ["vegas"]), "rush_+ opponent adjustment": ("ridge_opp", ["vegas"]),
             "rush_+ wind": ("ridge", ["vegas", "wind"]), "rush_+ play-by-play": ("ridge_pbp", ["vegas"])},
    "pass": {"pass_ridge": ("ridge", []), "pass_BASE (ridge+opponent)": ("ridge_opp", []), "pass_+ Vegas": ("ridge_opp", ["vegas"]),
             "pass_+ wind": ("ridge_opp", ["wind"]), "pass_+ play-by-play (module 1b)": ("plays_opp", [])},
}
BASE_OF = {"rush": "rush_BASE (ridge+Vegas)", "pass": "pass_BASE (ridge+opponent)"}
if "--round3" in sys.argv:    # passing only; the order of corrections is wind first (it is in the base), then whatever is added
    PIPES = {"rush": {}, "pass": {"pass_ridge": ("ridge", []), "pass_ridge + opponent": ("ridge_opp", []), "pass_BASE (ridge+opponent+wind)": ("ridge_opp", ["wind"]),
                                  "pass_+ Vegas": ("ridge_opp", ["wind", "vegas"]), "pass_+ play-by-play (module 1b)": ("plays_opp", ["wind"])}}
    BASE_OF = {"pass": "pass_BASE (ridge+opponent+wind)"}


def predict(G: pd.DataFrame, seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    teams = sorted(set(G.off_team))
    cfg, pcfg = Y.YardsConfig(), P.PlaysConfig(strength_volume=K, strength_efficiency=K)
    hist, out = [], []
    for season in seasons:
        prev = G[G.season == season - 1]
        prev_lg = {s: float(prev[f"{s}_yds"].mean()) for s in SIDES} if len(prev) else None
        prev_avg = prev.groupby("off_team")[[f"{s}_yds" for s in SIDES]].mean() if len(prev) else None
        prev_att = {s: prev.groupby("off_team")[f"{s}_att"].mean() for s in SIDES} if len(prev) else None
        prev_ypp = {s: prev.groupby("off_team")[f"{s}_yds_pbp"].sum() / prev.groupby("off_team")[f"{s}_att"].sum() for s in SIDES} if len(prev) else None
        pyd = lambda d: d[["off_team", "def_team", "rush_yds", "pass_yds"]]
        ppl = lambda d: d.assign(rush_yds=d.rush_yds_pbp, pass_yds=d.pass_yds_pbp)[["off_team", "def_team", "rush_att", "pass_att", "rush_yds", "pass_yds"]]
        prior = Y.build_prior(pyd(prev), teams, cfg) if len(prev) else None
        pprior = P.build_prior(ppl(prev), teams) if len(prev) else None
        for week in sorted(G.loc[G.season == season, "week"].unique()):
            cur, todo = G[(G.season == season) & (G.week < week)], G[(G.season == season) & (G.week == week)].copy()
            lg = {s: (float(cur[f"{s}_yds"].mean()) if len(cur) else (prev_lg[s] if prev_lg else np.nan)) for s in SIDES}
            if any(np.isnan(v) for v in lg.values()):
                continue
            ref = cur if len(cur) else prev
            l_att = {s: float(ref[f"{s}_att"].mean()) for s in SIDES}
            l_ypp = {s: float(ref[f"{s}_yds_pbp"].sum() / ref[f"{s}_att"].sum()) for s in SIDES}
            n_own = cur.groupby("off_team").size()
            full = Y.fit_ratings(pyd(cur), teams, cfg, prior, ridge=True, fallback_mu=prev_lg)
            plays_fit = P.fit_ratings(ppl(cur), teams, pcfg, pprior, ridge=True, fallback_mu=pprior.mu) if pprior is not None else None   # no play-level model in the first season
            bases = {}
            for s in SIDES:
                n = todo.off_team.map(n_own).fillna(0.0)
                own = cur.groupby("off_team")[f"{s}_yds"].mean()
                pa = todo.off_team.map(prev_avg[f"{s}_yds"]) if prev_avg is not None else pd.Series(np.nan, index=todo.index)
                avg_or_lg = todo.off_team.map(own).fillna(lg[s])
                bases[(s, "ridge")] = ((n * avg_or_lg + K * (0.3 * pa.fillna(lg[s]) + 0.7 * lg[s])) / (n + K)).to_numpy(float)
                bases[(s, "ridge_opp")] = np.array([full.expected(o, d)[s] for o, d in zip(todo.off_team, todo.def_team)])
                bases[(s, "plays_opp")] = np.array([plays_fit.expected(o, d)[f"{s}_yds"] for o, d in zip(todo.off_team, todo.def_team)]) if plays_fit is not None else np.full(len(todo), np.nan)
                att_m, att_s, yds_s = cur.groupby("off_team")[f"{s}_att"].mean(), cur.groupby("off_team")[f"{s}_att"].sum(), cur.groupby("off_team")[f"{s}_yds_pbp"].sum()
                pat = todo.off_team.map(prev_att[s]) if prev_att else pd.Series(np.nan, index=todo.index)
                ppp = todo.off_team.map(prev_ypp[s]) if prev_ypp else pd.Series(np.nan, index=todo.index)
                att_x = (n * todo.off_team.map(att_m).fillna(l_att[s]) + K * (0.3 * pat.fillna(l_att[s]) + 0.7 * l_att[s])) / (n + K)
                ypp_x = (todo.off_team.map(yds_s).fillna(0.0) + K * l_att[s] * (0.3 * ppp.fillna(l_ypp[s]) + 0.7 * l_ypp[s])) / (todo.off_team.map(att_s).fillna(0.0) + K * l_att[s])
                bases[(s, "ridge_pbp")] = (att_x * ypp_x).to_numpy(float)
            train = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
            train = train[train.season.isin([season - 1, season])] if len(train) else train
            fav_tr = train.fav.to_numpy(float) if len(train) else np.array([])
            ok = train.wind5.notna().to_numpy() if len(train) else np.array([], bool)
            wmean = float(train.wind5[ok].mean()) if ok.any() else 0.0
            x_now = {"vegas": todo.fav.to_numpy(float), "wind": (todo.wind5.fillna(wmean) - wmean).to_numpy(float)}
            x_tr = {"vegas": fav_tr, "wind": (train.wind5.to_numpy(float) - wmean) if len(train) else np.array([])}
            slopes = {}
            for s in SIDES:
                for name, (base_kind, corrs) in PIPES[s].items():
                    cur_pred = bases[(s, base_kind)]
                    todo[f"{name}__s0"] = cur_pred
                    for k, c in enumerate(corrs):
                        if len(train):
                            res = (train[f"{s}_yds"] - train[f"{name}__s{k}"]).to_numpy(float)
                            xs, m = x_tr[c], (np.ones(len(train), bool) if c == "vegas" else ok)
                            m = m & np.isfinite(res)          # training rows where this pipeline had no prediction (e.g. no play-level model in 2024) are skipped
                            den = float((xs[m] ** 2).sum())
                            slope = float((xs[m] * res[m]).sum() / den) if den else 0.0
                        else:
                            slope = 0.0
                        slopes[(name, c)] = slope
                        cur_pred = np.maximum(cur_pred + slope * x_now[c], 0.0)
                        todo[f"{name}__s{k + 1}"] = cur_pred
                    todo[name] = cur_pred
            for (name, c), v in slopes.items():
                todo[f"slope__{name}__{c}"] = v
            out.append(todo)
            keep = ["season", "week", "fav", "wind5", "rush_yds", "pass_yds"] + [c for c in todo.columns if "__s" in c and not c.startswith("slope")]
            hist.append(todo[keep])
    return pd.concat(out, ignore_index=True)


def table(bt: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = {}
    a = bt[f"{side}_yds"].to_numpy(float)
    base = bt[BASE_OF[side]].to_numpy(float)
    idx = np.random.default_rng(0).integers(0, len(a), (3000, len(a)))
    var = ((a - a.mean()) ** 2).mean()
    for name in PIPES[side]:
        m = metrics(bt, side, name)
        d = (a - base) ** 2 - (a - bt[name].to_numpy(float)) ** 2
        b = d[idx].mean(1) / var
        label = name.split("_", 1)[1]
        rows[label] = dict(R2=m["R2"], MAE=m["MAE"], bias=m["bias"], corr=m["corr"], pairs=m["pairs"], dR2=d.mean() / var, lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5))
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 230
    G = load_all()
    Pdf = predict(G)
    bt = Pdf[Pdf.season.isin([2025, 2026])]
    print("dR2 = change in R2 against the BASE of each side, with a 95% interval (positive = better)\n")
    for label, sub in ((f"weeks {MAIN_FROM}+", bt[bt.week >= MAIN_FROM]), (f"weeks 1-{MAIN_FROM - 1}", bt[bt.week < MAIN_FROM])):
        for side in [x for x in SIDES if PIPES[x]]:
            print(f"=== {side.upper()} YARDS, {label} ({len(sub)} team-games)")
            print(table(sub, side).to_string()); print()
    cols = [c for c in bt.columns if c.startswith("slope__")]
    print("fitted slopes at the end of 2025 and 2026 (Vegas: yards per point favoured; wind: yards per 5 mph above average):")
    print(bt.groupby("season")[cols].last().T.rename(index=lambda i: i.replace("slope__", "")).round(2).to_string())
