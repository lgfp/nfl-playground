"""Player touchdown models, walk-forward on 2025-26 (previous season at most): split a team's expected TDs among its players.

For each team-game and kind (rushing / receiving), the candidates are the players with real usage in the team's last 6 games. History is the team's games
since the previous season, weighted decay**age. Models turn that history plus the team's expected TDs (best team model) into expected TDs per player.

    uv run python -m analysis.td_player
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import td_data as D
from analysis import td_team as TT
from analysis.team_yards_backtest import load_env
from analysis.yards_backtest import load
from nflfp import team_yards as TY

TEAM_MODEL = "GLM implied+yards+xTD"
DECAYS = {"d85": 0.85, "d95": 0.95}
MIN_RECENT = {"rush": 8, "rec": 12}          # carries / targets in the last 6 team games to count as a candidate
GL = {"rush": 5, "rec": 10}                  # "goal-line" distance in yards


def player_games(plays: pd.DataFrame, kind: str) -> pd.DataFrame:
    p = plays[plays.play_type == ("run" if kind == "rush" else "pass")]
    pid, name = ("rusher_player_id", "rusher_player_name") if kind == "rush" else ("receiver_player_id", "receiver_player_name")
    p = p[p[pid].notna()].assign(pid=lambda x: x[pid], gl=lambda x: (x.yardline_100 <= GL[kind]).astype(float), rz=lambda x: (x.yardline_100 <= 20).astype(float))
    g = p.groupby(["season", "week", "posteam", "pid"], as_index=False).agg(name=(name, "last"), opps=("td", "size"), td=("td", "sum"), xtd=("xtd", "sum"), gl=("gl", "sum"), rz=("rz", "sum"))
    return g.rename(columns={"posteam": "team"})


def build(plays: pd.DataFrame, kind: str, team_pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, candidate): features from before the game + the actual TDs."""
    PG = player_games(plays, kind)
    qbs = set(plays[plays.play_type == "pass"].groupby("passer_player_id").size().loc[lambda s: s >= 40].index)
    order = PG[["season", "week", "team"]].drop_duplicates().sort_values(["team", "season", "week"])
    games = {t: list(zip(g.season, g.week)) for t, g in order.groupby("team")}
    by_team = {t: g for t, g in PG.groupby("team")}
    pred = team_pred.set_index(["season", "week", "team"])
    rows = []
    for t, gl in games.items():
        pg = by_team[t]
        for i, (s, w) in enumerate(gl):
            hist = [g for g in gl[:i] if g[0] >= s - 1]
            if not hist or (s, w) not in pred.index:
                continue
            h = pg.merge(pd.DataFrame(hist, columns=["season", "week"]), on=["season", "week"])
            age = {g: len(hist) - 1 - j for j, g in enumerate(hist)}
            h["age"] = [age[(a, b)] for a, b in zip(h.season, h.week)]
            last6 = h[h.age < 6].groupby("pid").opps.sum()
            cands = last6[last6 >= MIN_RECENT[kind]].index
            if len(cands) == 0:
                continue
            cur_actual = pg[(pg.season == s) & (pg.week == w)].set_index("pid")
            base = dict(season=s, week=w, team=t, team_exp=float(pred.loc[(s, w, t), f"{TEAM_MODEL}|{'rush' if kind == 'rush' else 'pass'}"]))
            feats = {}
            for dn, d in DECAYS.items():
                wt = np.power(d, h.age.to_numpy(float))
                hh = h.assign(wt=wt, **{c: h[c] * wt for c in ("opps", "td", "xtd", "gl", "rz")})
                tot = hh[["opps", "td", "xtd", "gl", "rz"]].sum()
                pl = hh.groupby("pid")[["opps", "td", "xtd", "gl", "rz"]].sum()
                neff = float(sum(np.power(d, a) for a in range(len(hist))))
                feats[dn] = (pl, tot, neff)
            for pid in cands:
                r = dict(base, pid=pid, name=h[h.pid == pid].name.iloc[-1], qb=float(pid in qbs),
                         td=float(cur_actual.td.get(pid, 0.0)), played=float(pid in cur_actual.index))
                for dn, (pl, tot, neff) in feats.items():
                    x = pl.loc[pid] if pid in pl.index else pd.Series(0.0, index=pl.columns)
                    r[f"{dn}_n"] = neff
                    for c in ("opps", "td", "xtd", "gl", "rz"):
                        r[f"{dn}_{c}"] = float(x[c]); r[f"{dn}_t{c}"] = float(tot[c])
                rows.append(r)
    return pd.DataFrame(rows)


def models(R: pd.DataFrame, kind: str) -> pd.DataFrame:
    P = R[["season", "week", "team", "pid", "name", "td", "played", "team_exp", "qb"]].copy()
    sh = lambda a, b, fb=None: np.where(R[b] > 0, R[a] / R[b].where(R[b] > 0, 1), 0.0 if fb is None else fb)
    for dn in DECAYS:
        vol = pd.Series(sh(f"{dn}_opps", f"{dn}_topps"), index=R.index)
        xs = pd.Series(sh(f"{dn}_xtd", f"{dn}_txtd", vol), index=R.index)
        gls = pd.Series(sh(f"{dn}_gl", f"{dn}_tgl", vol), index=R.index)
        rzs = pd.Series(sh(f"{dn}_rz", f"{dn}_trz", vol), index=R.index)
        tds = pd.Series(sh(f"{dn}_td", f"{dn}_ttd", vol), index=R.index)
        for k in (2.0, 4.0, 8.0):
            tds_r = (R[f"{dn}_td"] + k * vol * 1.0) / (R[f"{dn}_ttd"] + k)       # TD share shrunk toward volume share
            P[f"[{dn}] TD share (ridge {int(k)} TDs) x team"] = tds_r * R.team_exp
        P[f"[{dn}] volume share x team"] = vol * R.team_exp
        P[f"[{dn}] xTD share x team"] = xs * R.team_exp
        P[f"[{dn}] goal-line share x team"] = gls * R.team_exp
        P[f"[{dn}] red-zone share x team"] = rzs * R.team_exp
        P[f"[{dn}] mean(vol, xTD, goal-line) x team"] = (vol + xs + gls) / 3 * R.team_exp
        P[f"[{dn}] own xTD per game"] = R[f"{dn}_xtd"] / R[f"{dn}_n"]
        P[f"[{dn}] own TDs per game, ridge 8"] = (R[f"{dn}_td"] + 8 * 0.1) / (R[f"{dn}_n"] + 8)
        P[f"_vol_{dn}"], P[f"_xs_{dn}"], P[f"_gl_{dn}"], P[f"_rz_{dn}"] = vol, xs, gls, rzs
    return P


GLM = {"GLM team+xTD share": ["lt", "lx"], "GLM team+volume share": ["lt", "lv"], "GLM team+xTD+goal-line": ["lt", "lx", "lg"],
       "GLM team+vol+xTD+goal-line+rz": ["lt", "lv", "lx", "lg", "lr"], "GLM team+vol+xTD+goal-line+QB": ["lt", "lv", "lx", "lg", "qb"]}


def add_glm(P: pd.DataFrame, dn: str = "d85") -> pd.DataFrame:
    eps = 0.01
    X = dict(lt=np.log(P.team_exp.clip(lower=0.05)), lv=np.log(P[f"_vol_{dn}"] + eps), lx=np.log(P[f"_xs_{dn}"] + eps), lg=np.log(P[f"_gl_{dn}"] + eps),
             lr=np.log(P[f"_rz_{dn}"] + eps), qb=P.qb)
    y = P.td.to_numpy(float)
    for name, cols in GLM.items():
        M = np.column_stack([X[c].to_numpy(float) for c in cols])
        out = np.full(len(P), np.nan)
        for s in sorted(P.season.unique()):
            for w in sorted(P[P.season == s].week.unique()):
                te = ((P.season == s) & (P.week == w)).to_numpy()
                tr = (((P.season == s) & (P.week < w)) | (P.season == s - 1)).to_numpy()
                tr = tr & np.isfinite(M).all(1)
                if tr.sum() < 300:
                    continue
                b = D.poisson_fit(M[tr], y[tr], lam=2.0)
                out[te] = D.poisson_predict(b, M[te])
        P[f"[{dn}] {name}"] = out
    # constant: mean TD of all earlier candidate rows (walk-forward)
    const = np.full(len(P), np.nan)
    for s in sorted(P.season.unique()):
        for w in sorted(P[P.season == s].week.unique()):
            te = ((P.season == s) & (P.week == w)).to_numpy()
            tr = (((P.season == s) & (P.week < w)) | (P.season == s - 1)).to_numpy()
            if tr.sum() >= 300:
                const[te] = y[tr].mean()
    P["constant (mean candidate)"] = const
    return P


def star_flag(P: pd.DataFrame, kind: str, top: int) -> np.ndarray:
    """Top `top` candidates league-wide by decayed xTD per game, each week."""
    sc = P["[d85] own xTD per game"]
    rank = sc.groupby([P.season, P.week]).rank(ascending=False, method="first")
    return (rank <= top).to_numpy()


def report(P: pd.DataFrame, mask: np.ndarray, label: str, base="constant (mean candidate)"):
    model_cols = [c for c in P.columns if (c.startswith("[") or c.startswith("constant")) and np.isfinite(P[c]).any()]
    ok = mask & np.all([np.isfinite(P[c]) for c in model_cols], axis=0)
    y = P.td.to_numpy(float)[ok]
    gk = pd.factorize((P.season.astype(str) + P.week.astype(str) + P.team)[ok])[0]
    base_e = (y - P[base].to_numpy(float)[ok]) ** 2
    rng = np.random.default_rng(1)
    rows = []
    for c in model_cols:
        p = np.maximum(P[c].to_numpy(float)[ok], 0.003)
        e = (y - p) ** 2
        d = np.bincount(gk, base_e - e, gk.max() + 1)
        bs = d[rng.integers(0, len(d), size=(1000, len(d)))].sum(1) / base_e.sum()
        agg = pd.DataFrame(dict(a=P.pid.to_numpy()[ok], p=p, y=y)).groupby("a").sum()
        rows.append(dict(model=c, MAE=np.abs(y - p).mean(), PoisDev=D.poisson_deviance(y, p), skill=d.sum() / base_e.sum(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         corr=np.corrcoef(p, y)[0, 1], season_corr=np.corrcoef(agg.p, agg.y)[0, 1], bias=(p - y).mean()))
    print(f"\n--- {label}: n={ok.sum()} player-games, {len(set(P.pid.to_numpy()[ok]))} players, mean TD/game {y.mean():.3f} ---")
    print(pd.DataFrame(rows).sort_values("skill", ascending=False).head(14).to_string(index=False))


if __name__ == "__main__":
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 220
    plays = D.with_xtd(D.load_plays())
    T = D.team_games(plays)
    games, _ = load()
    F = TT.features(T, TY.walk_forward(games, load_env(), list(D.SEASONS)), D.environment())
    TP = TT.predict(F)
    for kind in ("rush", "rec"):
        R = build(plays, kind, TP)
        P = add_glm(models(R, kind))
        ev = ((P.week >= 4) & P.season.isin([2025, 2026])).to_numpy()
        P.to_pickle(f"/tmp/td_P_{kind}.pkl")
        print(f"\n################ {kind.upper()} TDs per player-game (candidates: {R.shape[0]} rows). skill = MSE reduction vs the mean candidate TD rate ################")
        report(P, ev, f"{kind}: all candidates")
        report(P, ev & star_flag(P, kind, 15 if kind == "rush" else 30), f"{kind}: stars (top {15 if kind == 'rush' else 30} by xTD/game each week)")
        report(P, ev & (P.played.to_numpy() == 1), f"{kind}: only games the player played")
