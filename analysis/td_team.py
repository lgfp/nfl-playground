"""Team touchdown models, walk-forward on 2025-26 (previous season at most): which predictor best forecasts a team's rushing and passing TDs in a game?

    uv run python -m analysis.td_team
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import td_data as D
from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import team_yards as TY

SIDES = ("rush", "pass")
K = 12.0          # pseudo-games for the feature ridges


def features(T: pd.DataFrame, yards: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """For every team-game: what was knowable before it (current season's earlier weeks + previous season) next to what happened."""
    rows = []
    for s in sorted(T.season.unique()):
        prev = T[T.season == s - 1]
        for w in sorted(T[T.season == s].week.unique()):
            cur = T[(T.season == s) & (T.week < w)]
            todo = T[(T.season == s) & (T.week == w)].copy()
            if cur.empty and prev.empty:
                continue
            base = cur if len(cur) else prev
            for side in SIDES:
                for col, name in ((f"{side}_td", "td"), (f"{side}_x", "x"), (f"{side}_yds", "y")):
                    lg = float(base[col].mean())
                    todo[f"lg_{name}_{side}"] = lg
                    pm = prev.groupby("team")[col].mean() if len(prev) else pd.Series(dtype=float)
                    pd_ = prev.groupby("opp")[col].mean() if len(prev) else pd.Series(dtype=float)
                    for w_prior, tag in ((0.3, ""), (0.0, "0")):
                        t_own = (w_prior * todo.team.map(pm).fillna(lg) + (1 - w_prior) * lg).to_numpy(float)
                        t_def = (w_prior * todo.opp.map(pd_).fillna(lg) + (1 - w_prior) * lg).to_numpy(float)
                        own_s, own_n = cur.groupby("team")[col].sum(), cur.groupby("team")[col].size()
                        def_s, def_n = cur.groupby("opp")[col].sum(), cur.groupby("opp")[col].size()
                        for k in ((4.0, 8.0, 16.0, 32.0) if (name == "td" and not tag) else (K,)):
                            ks = "" if k == K else f"{int(k)}"
                            todo[f"own{tag}{ks}_{name}_{side}"] = D.ridge_mean(todo.team.map(own_s).fillna(0.0).to_numpy(float), todo.team.map(own_n).fillna(0.0).to_numpy(float), k, t_own)
                            todo[f"def{tag}{ks}_{name}_{side}"] = D.ridge_mean(todo.opp.map(def_s).fillna(0.0).to_numpy(float), todo.opp.map(def_n).fillna(0.0).to_numpy(float), k, t_def)
                # team TD per yard (ridge toward the league rate with K games of league yards)
                own_td, own_y = cur.groupby("team")[f"{side}_td"].sum(), cur.groupby("team")[f"{side}_yds"].sum()
                lr = float(base[f"{side}_td"].sum() / base[f"{side}_yds"].sum())
                todo[f"lg_rate_{side}"] = lr
                todo[f"tmrate_{side}"] = ((todo.team.map(own_td).fillna(0.0) + K * todo[f"lg_td_{side}"] * 1.0) / (todo.team.map(own_y).fillna(0.0) + K * todo[f"lg_y_{side}"])).to_numpy(float)
            rows.append(todo)
    F = pd.concat(rows, ignore_index=True)
    F = F.merge(yards.rename(columns={"off_team": "team", "def_team": "opp"})[["season", "week", "team", "opp", "rush_exp", "pass_exp"]], on=["season", "week", "team", "opp"], how="left")
    return F.merge(env[["season", "week", "team", "fav", "implied"]], on=["season", "week", "team"], how="left")


def glm_features(F: pd.DataFrame, side: str) -> dict[str, np.ndarray]:
    lg_td, lg_x, lg_y = F[f"lg_td_{side}"], F[f"lg_x_{side}"], F[f"lg_y_{side}"]
    return {
        "imp": ((F.implied - 22.0) / 5.0).to_numpy(float),
        "yds": np.log(F[f"{side}_exp"].clip(lower=5) / lg_y).to_numpy(float),
        "tm": np.log(F[f"own16_td_{side}"] / lg_td).to_numpy(float),
        "def": np.log(F[f"def16_td_{side}"] / lg_td).to_numpy(float),
        "xt": np.log(F[f"own_x_{side}"] / lg_x).to_numpy(float),
        "dxt": np.log(F[f"def_x_{side}"] / lg_x).to_numpy(float),
    }


GLMS = {"GLM implied": ["imp"], "GLM yards": ["yds"], "GLM implied+yards": ["imp", "yds"], "GLM xTD own+opp": ["xt", "dxt"], "GLM TD own+opp": ["tm", "def"],
        "GLM implied+xTD": ["imp", "xt"], "GLM implied+yards+xTD": ["imp", "yds", "xt"], "GLM all six": ["imp", "yds", "tm", "def", "xt", "dxt"]}


def predict(F: pd.DataFrame) -> pd.DataFrame:
    P = F[["season", "week", "team", "opp"]].copy()
    for side in SIDES:
        lg, xl = F[f"lg_td_{side}"], F[f"lg_x_{side}"]
        P[f"league|{side}"] = lg
        for k in ("4", "8", "16", "32"):
            P[f"team ridge K={k} (30% last yr)|{side}"] = F[f"own{k}_td_{side}"]
        P[f"team ridge K=12 (no last yr)|{side}"] = F[f"own0_td_{side}"]
        P[f"opp-adjusted team K=16|{side}"] = F[f"own16_td_{side}"] * F[f"def16_td_{side}"] / lg
        P[f"opponent only K=16|{side}"] = F[f"def16_td_{side}"]
        P[f"yards x league rate|{side}"] = F[f"{side}_exp"] * F[f"lg_rate_{side}"]
        P[f"yards x team rate|{side}"] = F[f"{side}_exp"] * F[f"tmrate_{side}"]
        P[f"team xTD (own)|{side}"] = F[f"own_x_{side}"] * lg / xl
        P[f"team xTD opp-adjusted|{side}"] = F[f"own_x_{side}"] * F[f"def_x_{side}"] / xl * lg / xl
    # walk-forward fits that need training rows
    keys = F[["season", "week"]].to_numpy()
    out = {}
    for side in SIDES:
        feats = glm_features(F, side)
        y = F[f"{side}_td"].to_numpy(float)
        for name in list(GLMS) + ["OLS implied"]:
            out[f"{name}|{side}"] = np.full(len(F), np.nan)
        for s in sorted(F.season.unique()):
            for w in sorted(F[F.season == s].week.unique()):
                te = ((F.season == s) & (F.week == w)).to_numpy()
                tr = (((F.season == s) & (F.week < w)) | (F.season == s - 1)).to_numpy()
                for name, cols in GLMS.items():
                    X = np.column_stack([feats[c] for c in cols])
                    ok = tr & np.isfinite(X).all(1)
                    if ok.sum() < 150:
                        continue
                    b = D.poisson_fit(X[ok], y[ok])
                    okt = te & np.isfinite(X).all(1)
                    out[f"{name}|{side}"][okt] = D.poisson_predict(b, X[okt])
                imp = F.implied.to_numpy(float)
                ok = tr & np.isfinite(imp)
                if ok.sum() >= 150:
                    a1, a0 = np.polyfit(imp[ok], y[ok], 1)
                    okt = te & np.isfinite(imp)
                    out[f"OLS implied|{side}"][okt] = np.maximum(a0 + a1 * imp[okt], 0.05)
    for k, v in out.items():
        P[k] = v
    for side in SIDES:     # simple blends
        P[f"BLEND yards-rate + implied + xTD|{side}"] = P[[f"yards x team rate|{side}", f"OLS implied|{side}", f"team xTD (own)|{side}"]].mean(axis=1, skipna=False)
    return P


def report(F: pd.DataFrame, P: pd.DataFrame, from_week=4, seasons=(2025, 2026)):
    m = (F.week >= from_week) & F.season.isin(seasons)
    names = sorted({c.split("|")[0] for c in P.columns if "|" in c})
    gk = (F.season.astype(str) + "-" + F.week.astype(str) + "-" + np.minimum(F.team, F.opp) + np.maximum(F.team, F.opp)).to_numpy()
    codes = pd.factorize(gk[m.to_numpy()])[0]
    rng = np.random.default_rng(0)
    boots = rng.integers(0, codes.max() + 1, size=(1000, codes.max() + 1))
    results = {}
    for side in list(SIDES) + ["total"]:
        if side == "total":
            y = (F.rush_td + F.pass_td).to_numpy(float)
            pr = lambda n: P[f"{n}|rush"].to_numpy(float) + P[f"{n}|pass"].to_numpy(float)
        else:
            y = F[f"{side}_td"].to_numpy(float)
            pr = lambda n: P[f"{n}|{side}"].to_numpy(float)
        ok = m.to_numpy().copy()
        for n in names:
            ok &= np.isfinite(pr(n))
        yy = y[ok]
        base_e = (yy - pr("league")[ok]) ** 2
        gcodes = pd.factorize(gk[ok])[0]
        rows = []
        for n in names:
            p = np.maximum(pr(n)[ok], 0.02)
            e = (yy - p) ** 2
            d = base_e - e                                      # positive = better than the league average
            per_game = np.bincount(gcodes, d, gcodes.max() + 1)
            ng = len(per_game)
            bs = per_game[rng.integers(0, ng, size=(1000, ng))].sum(1) / base_e.sum()
            rows.append(dict(model=n, MAE=np.abs(yy - p).mean(), RMSE=np.sqrt(e.mean()), PoisDev=D.poisson_deviance(yy, p), bias=(p - yy).mean(),
                             skill=d.sum() / base_e.sum(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), corr=np.corrcoef(p, yy)[0, 1]))
        results[side] = (len(yy), yy.mean(), pd.DataFrame(rows).sort_values("skill", ascending=False))
    return results


if __name__ == "__main__":
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 200
    plays = D.with_xtd(D.load_plays())
    T = D.team_games(plays)
    games, _ = load()
    yards = TY.walk_forward(games, load_env(), list(D.SEASONS))
    F = features(T, yards, D.environment())
    P = predict(F)
    F.to_pickle("/tmp/td_F.pkl"); P.to_pickle("/tmp/td_P.pkl")
    for side, (n, mean, tab) in report(F, P).items():
        print(f"\n=== {side.upper()} TDs per team-game, weeks 4+ of 2025-26: n={n}, mean actual {mean:.3f}. skill = MSE reduction vs the league average to date (95% CI over games) ===")
        print(tab.to_string(index=False))
