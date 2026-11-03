from __future__ import annotations

import argparse

import pandas as pd

from . import data, lines as linemod, model, sleeper, weather


def _warn(adj, notes: list[str]) -> None:
    if adj is not None and adj.missing:
        print(f"\nnote: no betting line yet for {len(adj.missing)} team-games (left unadjusted) - lines are posted close to game day")
    if adj is not None and adj.missing_wind:
        print(f"note: no wind forecast for {len(adj.missing_wind)} team-games (treated as average wind)")
    for n in notes:
        print("note:", n)


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

    scoring = model.Scoring(ppr=a.ppr, pass_yard=a.pass_yard, pass_td=a.pass_td)
    pw = data.load_player_weeks(a.season, a.refresh)
    sched = data.load_schedule(a.season, a.refresh)

    week, matchups = data.next_matchups(sched, a.week)
    if a.week:  # explicit week: learn only from games before it
        pw = pw[pw["week"] < week]
    unknown = {t for pair in matchups for t in pair} - set(pw["team"])
    if unknown:
        raise SystemExit(f"Teams in schedule but not in stats (abbreviation mismatch?): {sorted(unknown)}")

    ratings = model.fit_ratings(model.team_games(pw), neutral=None if a.neutral is None else tuple(a.neutral))
    adjust = adj = None
    notes = []
    if not a.no_lines:
        use = tuple(a.use_lines) if a.use_lines else linemod.FEATURES
        wx = None
        if not a.no_wind:
            try:
                wx = weather.game_weather(sched)
            except Exception as e:  # network / rate limit: carry on without wind rather than fail
                notes.append(f"wind left out: weather unavailable ({e})")
        adj = linemod.fit_line_adjustment(model.team_games(pw), ratings, scoring, linemod.team_lines(sched, wx), use, wind=wx is not None)
        adjust = lambda offense, rush, rec: adj.adjust(week, offense, rush, rec)  # noqa: E731
        print("Game-environment adjustment (" + ", ".join(sorted(set(adj.use["rush"]) | set(adj.use["rec"]))) + "): " +
              "; ".join(f"{k} " + " ".join(f"{c:+.2f}" for c in v) for k, v in adj.coef.items()))
    print(f"Season {a.season}: learned from weeks {int(pw['week'].min())}-{int(pw['week'].max())}, projecting week {week}\n")
    for kind in ("run", "pass"):
        print(f"Defense vs the {kind}, worst first (per game; adj = against an average offense):")
        print(model.defense_ranking(ratings, scoring, kind).to_string())
        print()

    if a.teams:
        print(f"Expected team rushing + receiving points, week {week} (ppr={a.ppr}):")
        print(model.project_teams(ratings, matchups, scoring, not a.no_offense_adj, a.sort_by, adjust).to_string())
        _warn(adj, notes)
        return

    exclude, out = set(), None
    if not a.no_injuries:
        try:
            known = pw.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
            out = sleeper.unavailable(sleeper.load_players(a.refresh_injuries), known)
            exclude = set(out["player_id"])
        except Exception as e:  # network / rate limit: carry on with everyone available rather than fail
            notes.append(f"injuries not applied: Sleeper unavailable ({e})")
    shares = model.usage_shares(pw, a.window, exclude)
    if out is not None:
        usage = pw[pw["player_id"].isin(exclude)].groupby("player_id")[["carries", "targets", "attempts"]].sum().sum(axis=1)
        playing = {t for pair in matchups for t in pair}
        rel = out[out["player_id"].isin(usage[usage > 0].index) & out["team"].isin(playing)].copy()
        rel["usage"] = rel["player_id"].map(usage)
        top = rel.sort_values("usage", ascending=False).head(12)
        print(f"Sleeper injuries: dropped {len(rel)} Out/IR players with usage this season from the carry/target shares "
              f"(biggest: " + ", ".join(f"{r.name} {r.team} {r.status}" for r in top.itertuples()) + ")")
    proj = model.project_players(ratings, shares, matchups, scoring, not a.no_offense_adj, adjust)
    print(f"\nTop {a.top} projected players, week {week} (base fantasy points, ppr={a.ppr}):")
    print(proj.head(a.top).to_string())
    if a.csv:
        proj.to_csv(a.csv, index=False)
        print(f"\nWrote {len(proj)} rows to {a.csv}")
    _warn(adj, notes)


if __name__ == "__main__":
    main()
