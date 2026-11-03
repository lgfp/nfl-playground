"""The full projection pipeline as one call, so the CLI and other tools (e.g. the league tool) share it."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data, lines as linemod, model, sleeper, weather


@dataclass
class Result:
    season: int
    week: int
    matchups: list
    pw: pd.DataFrame                 # player-weeks the model learned from
    ratings: model.Ratings
    scoring: model.Scoring
    adjust: object                   # callable(offense, rush, rec) -> (rush, rec), or None
    adj: linemod.LineAdjustment | None
    shares: pd.DataFrame
    proj: pd.DataFrame               # player projections (one row per player with usage)
    injuries: pd.DataFrame | None    # Out/IR players Sleeper listed (None if not applied)
    adjustment_text: str = ""
    injury_text: str = ""
    notes: list[str] = field(default_factory=list)


def project(
    season: int = 2026, week: int | None = None, scoring: model.Scoring | None = None, *, refresh: bool = False,
    neutral: tuple | None = None, lines: bool = True, line_features: tuple | None = None, wind: bool = True,
    injuries: bool = True, refresh_injuries: bool = False, window: int | None = None, offense_adj: bool = True,
    players: bool = True, decay: float = model.DEFAULT_DECAY,
) -> Result:
    """Fit the model and project every player for `week` (default: the next unplayed week).

    `players=False` stops after the team-level setup (no usage shares, injuries or player projections).
    """
    scoring = scoring or model.Scoring()
    pw = data.load_player_weeks(season, refresh)
    sched = data.load_schedule(season, refresh)

    explicit = week is not None
    week, matchups = data.next_matchups(sched, week)
    if explicit:  # explicit week: learn only from games before it
        pw = pw[pw["week"] < week]
    unknown = {t for pair in matchups for t in pair} - set(pw["team"])
    if unknown:
        raise SystemExit(f"Teams in schedule but not in stats (abbreviation mismatch?): {sorted(unknown)}")

    ratings = model.fit_ratings(model.team_games(pw), neutral=neutral)
    res = Result(season=season, week=week, matchups=matchups, pw=pw, ratings=ratings, scoring=scoring,
                 adjust=None, adj=None, shares=pd.DataFrame(), proj=pd.DataFrame(), injuries=None)

    if lines:
        use = tuple(line_features) if line_features else linemod.FEATURES
        wx = None
        if wind:
            try:
                wx = weather.game_weather(sched)
            except Exception as e:  # network / rate limit: carry on without wind rather than fail
                res.notes.append(f"wind left out: weather unavailable ({e})")
        adj = linemod.fit_line_adjustment(model.team_games(pw), ratings, scoring, linemod.team_lines(sched, wx), use, wind=wx is not None)
        res.adj = adj
        res.adjust = lambda offense, rush, rec: adj.adjust(week, offense, rush, rec)
        res.adjustment_text = ("Game-environment adjustment (" + ", ".join(sorted(set(adj.use["rush"]) | set(adj.use["rec"]))) + "): " +
                               "; ".join(f"{k} " + " ".join(f"{c:+.2f}" for c in v) for k, v in adj.coef.items()))

    if not players:
        return res
    exclude = set()
    if injuries:
        try:
            known = pw.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
            res.injuries = sleeper.unavailable(sleeper.load_players(refresh_injuries), known)
            exclude = set(res.injuries["player_id"])
        except Exception as e:  # network / rate limit: carry on with everyone available rather than fail
            res.notes.append(f"injuries not applied: Sleeper unavailable ({e})")
    res.shares = model.usage_shares(pw, window, exclude, decay)
    if res.injuries is not None:
        usage = pw[pw["player_id"].isin(exclude)].groupby("player_id")[["carries", "targets", "attempts"]].sum().sum(axis=1)
        playing = {t for pair in matchups for t in pair}
        rel = res.injuries[res.injuries["player_id"].isin(usage[usage > 0].index) & res.injuries["team"].isin(playing)].copy()
        rel["usage"] = rel["player_id"].map(usage)
        top = rel.sort_values("usage", ascending=False).head(12)
        res.injury_text = (f"Sleeper injuries: dropped {len(rel)} Out/IR players with usage this season from the carry/target shares "
                           f"(biggest: " + ", ".join(f"{r.name} {r.team} {r.status}" for r in top.itertuples()) + ")")
    res.proj = model.project_players(ratings, res.shares, matchups, scoring, offense_adj, res.adjust)
    return res
