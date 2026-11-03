from __future__ import annotations

import argparse

import pandas as pd

from . import lines as linemod, model, pipeline


def _warn(res: pipeline.Result) -> None:
    adj = res.adj
    if adj is not None and adj.missing:
        print(f"\nnote: no betting line yet for {len(adj.missing)} team-games (left unadjusted) - lines are posted close to game day")
    if adj is not None and adj.missing_wind:
        print(f"note: no wind forecast for {len(adj.missing_wind)} team-games (treated as average wind)")
    for n in res.notes:
        print("note:", n)


def main() -> None:
    ap = argparse.ArgumentParser(prog="nflfp", description="Opponent-adjusted expected fantasy points per player.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="week to project (default: next unplayed week)")
    ap.add_argument("--ppr", type=float, default=0.5, help="points per reception (default 0.5)")
    ap.add_argument("--pass-yard", type=float, default=0.04, help="points per passing yard (default 0.04)")
    ap.add_argument("--pass-td", type=float, default=4.0, help="points per passing TD (default 4)")
    ap.add_argument("--window", type=int, help="only use each team's last N games for usage shares")
    ap.add_argument("--decay", type=float, default=model.DEFAULT_DECAY,
                    help="weight recent games more in the carry/target shares: a game k games back counts decay**k "
                         f"(default {model.DEFAULT_DECAY}; 1.0 = all games equal)")
    ap.add_argument("--top", type=int, default=30, help="rows to print")
    ap.add_argument("--no-offense-adj", action="store_true", help="skip the opposing-offense-vs-average adjustment")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--teams", action="store_true", help="print team totals (rushing + receiving) instead of players")
    ap.add_argument("--sort-by", default="rush_pts", choices=["rush_pts", "rec_pts", "total_pts"], help="sort for --teams (default rush_pts)")
    ap.add_argument("--neutral", nargs="*", default=None, choices=list(model.COMPONENTS),
                    help="components to leave at the league average (no opponent adjustment), e.g. rush_tds (default: rush_tds; pass --neutral with no values for none)")
    ap.add_argument("--no-lines", action="store_true", help="turn off the betting-line + wind adjustment (on by default)")
    ap.add_argument("--use-lines", nargs="*", choices=list(linemod.FEATURES), default=None,
                    help="which betting-line features to use (default: spread and total)")
    ap.add_argument("--no-wind", action="store_true", help="leave wind out of the receiving adjustment")
    ap.add_argument("--no-injuries", action="store_true",
                    help="don't drop players Sleeper lists as Out / IR from the carry and target shares (dropped by default)")
    ap.add_argument("--refresh-injuries", action="store_true", help="re-download Sleeper's player list (cached for 24h)")
    ap.add_argument("--csv", help="also write the full player projection to this CSV")
    a = ap.parse_args()
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 200

    res = pipeline.project(
        a.season, a.week, model.Scoring(ppr=a.ppr, pass_yard=a.pass_yard, pass_td=a.pass_td), refresh=a.refresh,
        neutral=None if a.neutral is None else tuple(a.neutral), lines=not a.no_lines, line_features=a.use_lines,
        wind=not a.no_wind, injuries=not a.no_injuries, refresh_injuries=a.refresh_injuries, window=a.window, decay=a.decay,
        offense_adj=not a.no_offense_adj, players=not a.teams)

    if res.adjustment_text:
        print(res.adjustment_text)
    pw = res.pw
    print(f"Season {a.season}: learned from weeks {int(pw['week'].min())}-{int(pw['week'].max())}, projecting week {res.week}\n")
    for kind in ("run", "pass"):
        print(f"Defense vs the {kind}, worst first (per game; adj = against an average offense):")
        print(model.defense_ranking(res.ratings, res.scoring, kind).to_string())
        print()

    if a.teams:
        print(f"Expected team rushing + receiving points, week {res.week} (ppr={a.ppr}):")
        print(model.project_teams(res.ratings, res.matchups, res.scoring, not a.no_offense_adj, a.sort_by, res.adjust).to_string())
        _warn(res)
        return

    if res.injury_text:
        print(res.injury_text)
    print(f"\nTop {a.top} projected players, week {res.week} (base fantasy points, ppr={a.ppr}):")
    print(res.proj.head(a.top).to_string())
    if a.csv:
        res.proj.to_csv(a.csv, index=False)
        print(f"\nWrote {len(res.proj)} rows to {a.csv}")
    _warn(res)


if __name__ == "__main__":
    main()
