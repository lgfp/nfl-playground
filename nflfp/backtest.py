"""Walk-forward backtest of the team-level rushing / receiving point predictions.

For each week w >= min_week the model is fit on weeks < w only, then predicts every team's
rushing and receiving points in week w's games; predictions are compared to what happened.
Baselines: the league average to date, and the team's own average to date.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import data, model


def team_points(pw: pd.DataFrame, s: model.Scoring) -> pd.DataFrame:
    tg = model.team_games(pw)
    tg["rush_pts"], tg["rec_pts"] = model.points(tg, s)
    return tg


def walk_forward(pw: pd.DataFrame, s: model.Scoring, min_week: int = 4) -> pd.DataFrame:
    """One row per (team, week) with actual, model, league-avg and team-avg predictions."""
    games = team_points(pw, s)
    rows = []
    for w in sorted(games["week"].unique()):
        if w < min_week:
            continue
        past, cur = games[games["week"] < w], games[games["week"] == w]
        r = model.fit_ratings(past)
        for g in cur.itertuples():
            if g.off_team not in r.off.index or g.def_team not in r.dfn.index:
                continue
            rush, rec = model.points(model.team_expectation(r, g.off_team, g.def_team), s)
            own = past[past["off_team"] == g.off_team]
            rows.append(dict(
                week=w, team=g.off_team, opp=g.def_team,
                act_rush=g.rush_pts, act_rec=g.rec_pts,
                mod_rush=rush, mod_rec=rec,
                lg_rush=past["rush_pts"].mean(), lg_rec=past["rec_pts"].mean(),
                own_rush=own["rush_pts"].mean(), own_rec=own["rec_pts"].mean(),
            ))
    return pd.DataFrame(rows)


def _metrics(act: pd.Series, pred: pd.Series) -> dict:
    err = pred - act
    ss_res, ss_tot = (err ** 2).sum(), ((act - act.mean()) ** 2).sum()
    return dict(MAE=err.abs().mean(), RMSE=np.sqrt((err ** 2).mean()), bias=err.mean(),
                R2=1 - ss_res / ss_tot, corr=np.corrcoef(act, pred)[0, 1])


def spread(bt: pd.DataFrame) -> pd.DataFrame:
    """Dispersion of the actual team-game outcomes (CV = sd / mean)."""
    d = {k: dict(mean=bt[f"act_{k}"].mean(), sd=bt[f"act_{k}"].std(), CV=bt[f"act_{k}"].std() / bt[f"act_{k}"].mean())
         for k in ("rush", "rec")}
    return pd.DataFrame(d).T


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for kind in ("rush", "rec"):
        for label, col in (("model", "mod"), ("team avg", "own"), ("league avg", "lg")):
            out[(kind, label)] = _metrics(bt[f"act_{kind}"], bt[f"{col}_{kind}"])
    return pd.DataFrame(out).T


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nflfp.backtest")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--min-week", type=int, default=4, help="first week to predict (needs history)")
    ap.add_argument("--ppr", type=float, default=0.5)
    ap.add_argument("--csv", help="write per team-game predictions here")
    a = ap.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    s = model.Scoring(ppr=a.ppr)

    frames = []
    for season in a.seasons:
        bt = walk_forward(data.load_player_weeks(season), s, a.min_week)
        bt.insert(0, "season", season)
        frames.append(bt)
    bt = pd.concat(frames, ignore_index=True)
    print(f"{len(bt)} team-games, seasons {a.seasons}, predicting weeks >= {a.min_week}, trained on earlier weeks only\n")
    print("Actual outcomes (points per team-game):")
    print(spread(bt).to_string())
    print("\nPrediction error (R2 > 0 means better than always guessing the mean):")
    print(summarize(bt).to_string())
    if a.csv:
        bt.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
