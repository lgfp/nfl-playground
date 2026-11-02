"""Backtest of the team-level rushing / receiving point predictions.

For each tested week w the model is fit on the rest of the season, then predicts every team's
rushing and receiving points in week w's games; predictions are compared to what happened.

* leave-one-week-out (default): train on *all* other weeks, past and future - the tested week is
  the only thing held out. Shows how well the model describes a week given a full season.
* walk-forward (--forward): train on earlier weeks only - what you could actually have known.

Baselines (same training weeks): the league average, and the team's own average.
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


def walk_forward(pw: pd.DataFrame, s: model.Scoring, min_week: int = 4, leave_one_out: bool = False, **fit_kwargs
) -> pd.DataFrame:
    """One row per (team, week) with actual, model, league-avg and team-avg predictions.

    leave_one_out=False trains on weeks before the tested week; True on every other week.
    """
    games = team_points(pw, s)
    rows = []
    for w in sorted(games["week"].unique()):
        if w < min_week:
            continue
        cur = games[games["week"] == w]
        past = games[games["week"] != w] if leave_one_out else games[games["week"] < w]
        r = model.fit_ratings(past, **fit_kwargs)
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
    ap.add_argument("--forward", action="store_true", help="train on earlier weeks only (default: all other weeks)")
    ap.add_argument("--min-week", type=int, help="first week to predict (default 1, or 4 with --forward)")
    ap.add_argument("--neutral", nargs="*", default=None, choices=list(model.COMPONENTS), help="components pinned to league average (default: rush_tds; no values = none)")
    ap.add_argument("--ppr", type=float, default=0.5)
    ap.add_argument("--csv", help="write per team-game predictions here")
    a = ap.parse_args()
    pd.options.display.float_format = "{:.3f}".format
    s = model.Scoring(ppr=a.ppr)
    min_week = a.min_week or (4 if a.forward else 1)
    how = "earlier weeks only" if a.forward else "all other weeks of the season"

    frames = []
    for season in a.seasons:
        bt = walk_forward(data.load_player_weeks(season), s, min_week, leave_one_out=not a.forward, neutral=None if a.neutral is None else tuple(a.neutral))
        bt.insert(0, "season", season)
        frames.append(bt)
    bt = pd.concat(frames, ignore_index=True)
    print(f"{len(bt)} team-games, seasons {a.seasons}, predicting weeks >= {min_week}, trained on {how}\n")
    print("Actual outcomes (points per team-game):")
    print(spread(bt).to_string())
    print("\nPrediction error (R2 > 0 means better than always guessing the mean):")
    print(summarize(bt).to_string())
    if a.csv:
        bt.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
