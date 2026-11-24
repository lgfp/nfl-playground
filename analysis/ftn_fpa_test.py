"""Do FTN's DVOA-adjusted fantasy points against (FPA) explain what Sleeper's projections miss? Weeks 9 and 10 of 2026.

The only FPA file is a season-to-date snapshot as of week 11, so it already contains the week 9-10 games being predicted. To see how much of any
signal is that leak, the same test runs on our own raw fantasy points allowed (per defense and position, per game, above the league average):
  own_leak   through week 11 - the same leak as FTN's file
  own_clean  only games before the week predicted - what could actually have been known
FTN beating own_leak = the opponent adjustment adds something; FTN ~ own_leak >> own_clean = the leak is the signal.

A player gets his share of the position's projection on his team (sl / team position total) times the defense's FPA for the position; FTN's WR1 / WR2 /
other-WR columns go to the team's WRs ranked by projection. The weight on that adjustment is fit on one week and applied to the other (2-fold).

    uv run python -m analysis.ftn_fpa_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import ftn, ml_data as MD
from analysis.ml_model import POOL_MIN, vs
from nflfp import data, sleeper
from nflfp.model import Scoring

WEEKS = (9, 10)
POS = ("QB", "RB", "WR", "TE")


def actual(season: int = 2026, sc: Scoring = Scoring()) -> pd.DataFrame:
    pw = data.load_player_weeks(season)
    pw = pw[pw.position.isin(POS)].copy()
    f = lambda c: pw[c].fillna(0)
    pw["pts"] = (sc.ppr * f("receptions") + sc.yard * (f("rushing_yards") + f("receiving_yards")) + sc.td * (f("rushing_tds") + f("receiving_tds"))
                 + sc.pass_yard * f("passing_yards") + sc.pass_td * f("passing_tds"))
    return pw.rename(columns={"opponent_team": "opp"})[["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts"]]


def own_fpa(act: pd.DataFrame, through: int) -> pd.DataFrame:
    """Per defense and position: points allowed per game to the position in weeks <= through, minus the league average per team-game."""
    g = act[act.week <= through].groupby(["week", "opp", "position"]).pts.sum().reset_index()
    per = g.groupby(["opp", "position"]).pts.mean().rename("fpa").reset_index()
    return per.assign(fpa=per.fpa - per.groupby("position").fpa.transform("mean"))


def ftn_fpa() -> pd.DataFrame:
    x = pd.read_csv(ftn._file("DVOA_adjusted_fantasy_points_against*.csv"))
    x["opp"] = x.DEFENSE.map(lambda c: ftn.TEAM.get(c, c))
    return x.drop(columns="DEFENSE")


def cross_fit(d: pd.DataFrame, x: str) -> pd.Series:
    """Projection + beta * x, beta (through the origin, all positions together) fit on the other week."""
    out = pd.Series(np.nan, index=d.index)
    r = d.pts - d.sl
    for w in WEEKS:
        tr, te = d.week != w, d.week == w
        beta = (d.loc[tr, x] * r[tr]).sum() / (d.loc[tr, x] ** 2).sum()
        out[te] = d.loc[te, "sl"] + beta * d.loc[te, x]
    return out


if __name__ == "__main__":
    act = actual()
    dump = sleeper.load_players()
    known = act.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    sl = pd.concat([MD.sleeper_points(2026, w, known, dump) for w in WEEKS], ignore_index=True)[["player_id", "season", "week", "sl"]]
    d = sl.merge(act, on=["player_id", "season", "week"], how="inner")
    d = d[d.sl >= POOL_MIN].reset_index(drop=True)
    tot = d.groupby(["week", "team", "position"]).sl.transform("sum")
    d["share"] = d.sl / tot
    d["wr_rank"] = d[d.position == "WR"].groupby(["week", "team"]).sl.rank(ascending=False, method="first")

    F = ftn_fpa()
    long = F.melt(id_vars="opp", value_vars=list(POS), var_name="position", value_name="ftn")
    d = d.merge(long, on=["opp", "position"], how="left")
    wr = F.set_index("opp")[["WR1", "WR2", "Other WR"]]
    col = d.wr_rank.map({1.0: "WR1", 2.0: "WR2"}).fillna("Other WR")
    d["ftn_wrsplit"] = np.where(d.position == "WR", [wr.at[o, c] if o in wr.index else np.nan for o, c in zip(d.opp, col)], d.ftn * d.share)
    d["ftn"] = d.ftn * d.share
    leak = own_fpa(act, 11).rename(columns={"fpa": "own_leak"})
    d = d.merge(leak, on=["opp", "position"], how="left")
    d["own_leak"] *= d.share
    d["own_clean"] = np.nan
    for w in WEEKS:
        cl = own_fpa(act, w - 1).set_index(["opp", "position"]).fpa
        m = d.week == w
        d.loc[m, "own_clean"] = [cl.get((o, p), np.nan) for o, p in zip(d.opp[m], d.position[m])]
    d["own_clean"] *= d.share
    d = d.dropna(subset=["ftn", "own_leak", "own_clean", "ftn_wrsplit"]).reset_index(drop=True)

    r = d.pts - d.sl
    print(f"2026 weeks {WEEKS}, {len(d)} player-games with a Sleeper projection >= {POOL_MIN}; Sleeper MAE {r.abs().mean():.3f}")
    print("FTN FPA vs our raw points allowed through week 11 (same as-of date), correlation by position:",
          {p: round(own_fpa(act, 11).merge(long, on=["opp", "position"]).query("position == @p")[["fpa", "ftn"]].corr().iloc[0, 1], 2) for p in POS})
    print("\ncorrelation of each adjustment with Sleeper's miss (actual - projection):")
    labels = {"ftn": "FTN FPA (as of wk 11: leaked)", "ftn_wrsplit": "FTN FPA, WR1/WR2/other split (leaked)",
              "own_leak": "our raw FPA thru wk 11 (same leak)", "own_clean": "our raw FPA, prior weeks only (clean)"}
    print(pd.DataFrame({lab: {p: np.corrcoef(d.loc[d.position == p, c], r[d.position == p])[0, 1] for p in POS} | {"all": np.corrcoef(d[c], r)[0, 1]}
                        for c, lab in labels.items()}).T.round(3).to_string())
    print("\nSleeper + beta x adjustment, beta fit on the other week (MAE change vs Sleeper; negative = information gain; 95% interval over team-weeks):")
    for c, lab in labels.items():
        d[f"p_{c}"] = cross_fit(d, c)
        x, lo, hi = vs(d, f"p_{c}", "sl")
        betas = [((d.loc[d.week != w, c] * r[d.week != w]).sum() / (d.loc[d.week != w, c] ** 2).sum()) for w in WEEKS]
        print(f"   {lab:40s} {x:+.3f} [{lo:+.3f}, {hi:+.3f}]   beta fit on wk10 / wk9: {betas[0]:.2f} / {betas[1]:.2f}")
