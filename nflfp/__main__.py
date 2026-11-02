from __future__ import annotations

import argparse

import pandas as pd

from . import data, model


def main() -> None:
    ap = argparse.ArgumentParser(prog="nflfp", description="Opponent-adjusted expected fantasy points per player.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="week to project (default: next unplayed week)")
    ap.add_argument("--ppr", type=float, default=0.5, help="points per reception (default 0.5)")
    ap.add_argument("--pass-yard", type=float, default=0.04, help="points per passing yard (default 0.04)")
    ap.add_argument("--pass-td", type=float, default=4.0, help="points per passing TD (default 4)")
    ap.add_argument("--window", type=int, help="only use each team's last N games for usage shares")
    ap.add_argument("--top", type=int, default=30, help="rows to print")
    ap.add_argument("--no-offense-adj", action="store_true", help="skip the opposing-offense-vs-average adjustment")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--csv", help="also write the full player projection to this CSV")
    a = ap.parse_args()
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 200

    scoring = model.Scoring(ppr=a.ppr, pass_yard=a.pass_yard, pass_td=a.pass_td)
    pw = data.load_player_weeks(a.season, a.refresh)
    sched = data.load_schedule(a.season, a.refresh)

    week, matchups = data.next_matchups(sched, a.week)
    if a.week:  # explicit week: learn only from games before it
        pw = pw[pw["week"] < week]
    unknown = {t for pair in matchups for t in pair} - set(pw["team"])
    if unknown:
        raise SystemExit(f"Teams in schedule but not in stats (abbreviation mismatch?): {sorted(unknown)}")

    ratings = model.fit_ratings(model.team_games(pw))
    print(f"Season {a.season}: learned from weeks {int(pw['week'].min())}-{int(pw['week'].max())}, projecting week {week}\n")
    print("Defenses (fantasy points allowed per game; adj = to an average offense), worst first:")
    print(model.defense_table(ratings, scoring).to_string())

    shares = model.usage_shares(pw, a.window)
    proj = model.project_players(ratings, shares, matchups, scoring, not a.no_offense_adj)
    print(f"\nTop {a.top} projected players, week {week} (base fantasy points, ppr={a.ppr}):")
    print(proj.head(a.top).to_string())
    if a.csv:
        proj.to_csv(a.csv, index=False)
        print(f"\nWrote {len(proj)} rows to {a.csv}")


if __name__ == "__main__":
    main()
