"""Playoff and championship odds for a fantasy league: the rest of the regular season and the playoff bracket, simulated.

Pure logic (no network): `best_lineup` picks a team's best projected starters, `parse_bracket` reads Sleeper's playoff bracket, `simulate` plays the season out many times.

Model: every team scores Normal(its best-lineup projection, sd) each week (floored at 0). Regular-season games are played from the league's schedule; teams are
ranked by wins (a tie counts half) and then points for, the top `n_playoff` make the playoffs, and the bracket (byes, one-week or two-week rounds, a third-place
game) is played with the same weekly scores. Outputs are per-team probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLEX_ELIGIBLE = {"FLEX": ("RB", "WR", "TE"), "SUPER_FLEX": ("QB", "RB", "WR", "TE"), "REC_FLEX": ("WR", "TE"), "WRRB_FLEX": ("RB", "WR")}
FIXED = ("QB", "RB", "WR", "TE", "K", "DEF")


# ------------------------------------------------------------------------------------------------------ points and lineups
def points_from_stats(stats: dict, scoring: dict) -> float:
    """Fantasy points of a projected stat line under a Sleeper league's `scoring_settings` (works for kickers and defenses too: bucket keys like
    `pts_allow_14_20` are 0/1 indicators in the projection feed)."""
    return float(sum(scoring[k] * v for k, v in stats.items() if k in scoring and isinstance(v, (int, float)) and k != "gp"))


def lineup_slots(roster_positions: list[str]) -> dict[str, int]:
    """Counts of starting slots from Sleeper's `roster_positions` (bench slots dropped). Raises for slot types this module cannot fill (IDP etc.)."""
    slots: dict[str, int] = {}
    for p in roster_positions:
        if p in ("BN", "IR", "TAXI"):
            continue
        if p not in FIXED and p not in FLEX_ELIGIBLE:
            raise ValueError(f"unsupported lineup slot {p!r}")
        slots[p] = slots.get(p, 0) + 1
    return slots


def best_lineup(players: list[tuple[str, str, float]], slots: dict[str, int]) -> tuple[float, list[str]]:
    """The highest-scoring legal lineup from (id, position, projected points): each fixed position takes its best, then the flex slots (most restrictive first)
    take the best of what is left. Returns (total, starter ids). Unfilled slots score 0."""
    by_pos: dict[str, list] = {}
    for pid, pos, pts in sorted(players, key=lambda t: -t[2]):
        by_pos.setdefault(pos, []).append((pid, pts))
    taken, total = [], 0.0
    for pos in FIXED:
        for pid, pts in by_pos.get(pos, [])[:slots.get(pos, 0)]:
            taken.append(pid)
            total += pts
    left = sorted([(pid, pts, pos) for pos, v in by_pos.items() for pid, pts in v[slots.get(pos, 0):] if pos in ("QB", "RB", "WR", "TE")], key=lambda t: -t[1])
    for flex in sorted((s for s in slots if s in FLEX_ELIGIBLE), key=lambda s: len(FLEX_ELIGIBLE[s])):
        for _ in range(slots[flex]):
            i = next((j for j, (pid, pts, pos) in enumerate(left) if pos in FLEX_ELIGIBLE[flex]), None)
            if i is not None:
                pid, pts, _ = left.pop(i)
                taken.append(pid)
                total += pts
    return total, taken


# ------------------------------------------------------------------------------------------------------------------ playoffs
def parse_bracket(winners_bracket: list[dict], seed_of_roster: dict[int, int]) -> list[dict]:
    """Sleeper's winners bracket -> matches `{id, round, place, a, b}` where a / b is ('seed', k) or ('w' | 'l', match id).
    Sleeper fills round one with the current standings' roster ids; `seed_of_roster` maps those back to seeds so the bracket can be replayed with simulated seeds."""
    def side(m, k):
        if m.get(k) is not None:
            return ("seed", seed_of_roster[m[k]])
        src = m[f"{k}_from"]
        (kind, mid), = src.items()
        return (kind, mid)
    return [dict(id=m["m"], round=m["r"], place=m.get("p"), a=side(m, "t1"), b=side(m, "t2")) for m in sorted(winners_bracket, key=lambda m: m["m"])]


def round_weeks(n_rounds: int, start_week: int, round_type: int = 0) -> dict[int, list[int]]:
    """Weeks each playoff round spans. Sleeper `playoff_round_type`: 0 one week per round, 1 the championship takes two weeks, 2 every round takes two."""
    out, w = {}, start_week
    for r in range(1, n_rounds + 1):
        two = round_type == 2 or (round_type == 1 and r == n_rounds)
        out[r] = [w, w + 1] if two else [w]
        w += len(out[r])
    return out


def standard_bracket(n: int) -> list[dict]:
    """A seed-based bracket for 2, 4 or 6 teams (6: seeds 1-2 get byes and the top bye plays the lowest remaining seed), with a third-place game."""
    if n == 2:
        return [dict(id=1, round=1, place=1, a=("seed", 1), b=("seed", 2))]
    if n == 4:
        return [dict(id=1, round=1, place=None, a=("seed", 1), b=("seed", 4)), dict(id=2, round=1, place=None, a=("seed", 2), b=("seed", 3)),
                dict(id=3, round=2, place=1, a=("w", 1), b=("w", 2)), dict(id=4, round=2, place=3, a=("l", 1), b=("l", 2))]
    if n == 6:
        return [dict(id=1, round=1, place=None, a=("seed", 3), b=("seed", 6)), dict(id=2, round=1, place=None, a=("seed", 4), b=("seed", 5)),
                dict(id=3, round=2, place=None, a=("seed", 1), b=("w", 2)), dict(id=4, round=2, place=None, a=("seed", 2), b=("w", 1)),
                dict(id=5, round=3, place=1, a=("w", 3), b=("w", 4)), dict(id=6, round=3, place=3, a=("l", 3), b=("l", 4))]
    raise ValueError(f"no standard bracket for {n} teams")


# ------------------------------------------------------------------------------------------------------------------ simulation
@dataclass
class Season:
    ids: list                                   # team labels (roster ids)
    wins: np.ndarray                            # wins so far (a tie counts 0.5)
    pf: np.ndarray                              # points scored so far
    schedule: dict[int, list[tuple[int, int]]]  # remaining regular-season week -> pairs of team indices
    mean: dict[int, np.ndarray]                 # week -> expected score of every team (regular-season and playoff weeks)
    n_playoff: int
    bracket: list[dict] = field(default_factory=list)
    rounds: dict[int, list[int]] = field(default_factory=dict)
    losses: np.ndarray | None = None


def simulate(s: Season, sd: float | np.ndarray = 21.0, n: int = 100000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = len(s.ids)
    weeks = sorted(s.mean)
    sdv = np.broadcast_to(np.asarray(sd, dtype=float), (T,))
    score = {w: np.maximum(rng.normal(s.mean[w], sdv, size=(n, T)), 0.0) for w in weeks}
    wins, pf = np.tile(np.asarray(s.wins, float), (n, 1)), np.tile(np.asarray(s.pf, float), (n, 1))
    for w, pairs in sorted(s.schedule.items()):
        sc = score[w]
        for a, b in pairs:
            win_a = (sc[:, a] > sc[:, b]).astype(float) + 0.5 * (sc[:, a] == sc[:, b])
            wins[:, a] += win_a
            wins[:, b] += 1 - win_a
        pf += sc
    order = np.argsort(-(wins * 1e6 + pf), axis=1)            # seeds: columns are seeds 1..T
    rank = np.argsort(order, axis=1)                          # rank[i, t] = seed index (0-based) of team t
    k = s.n_playoff
    res = pd.DataFrame({"team": s.ids})
    res["exp_wins"] = wins.mean(axis=0)
    res["p_playoffs"] = (rank < k).mean(axis=0)
    for j in range(k):
        res[f"p_seed{j + 1}"] = (rank == j).mean(axis=0)
    res["p_final"], res["p_champion"], res["p_third"] = 0.0, 0.0, 0.0
    if s.bracket:
        winner, loser = {}, {}
        reach_final, champ, third = np.zeros((n, T), bool), np.zeros((n, T), bool), np.zeros((n, T), bool)
        rows = np.arange(n)

        def team(spec):
            kind, v = spec
            if kind == "seed":
                return order[:, v - 1]
            return winner[v] if kind == "w" else loser[v]

        for m in sorted(s.bracket, key=lambda m: (m["round"], m["id"])):
            a, b = team(m["a"]), team(m["b"])
            ws = s.rounds[m["round"]]
            sa = sum(score[w][rows, a] for w in ws)
            sb = sum(score[w][rows, b] for w in ws)
            a_wins = sa >= sb
            winner[m["id"]], loser[m["id"]] = np.where(a_wins, a, b), np.where(a_wins, b, a)
            if m["place"] == 1:
                reach_final[rows, a] = True
                reach_final[rows, b] = True
                champ[rows, winner[m["id"]]] = True
            elif m["place"] == 3:
                third[rows, winner[m["id"]]] = True
        res["p_final"], res["p_champion"], res["p_third"] = reach_final.mean(axis=0), champ.mean(axis=0), third.mean(axis=0)
    return res


def clinch_status(s: Season, max_games: int = 22) -> list[str] | None:
    """Exact (not simulated) playoff status of every team from the win/loss outcomes of all remaining regular-season games (draws ignored):
    'clinched' (in even if every tie on wins goes against the team), 'eliminated' (out even if every tie goes its way), 'alive (points tiebreak only)'
    (cannot make it on wins alone in any outcome, so only a tiebreak on points for could save it) or 'alive'. None if more than `max_games` remain."""
    games = [(a, b) for _, ps in sorted(s.schedule.items()) for a, b in ps]
    G, T, k = len(games), len(s.ids), s.n_playoff
    if G > max_games:
        return None
    lo_more, hi_geq, lo_geq = np.full(T, T), np.zeros(T, int), np.full(T, T)
    chunk = 1 << min(G, 16)
    for start in range(0, 1 << G, chunk):
        idx = np.arange(start, min(start + chunk, 1 << G))
        bits = (idx[:, None] >> np.arange(G)) & 1                       # 1: the first team of the game wins
        w = np.tile(np.asarray(s.wins, float), (len(idx), 1))
        for g, (a, b) in enumerate(games):
            w[:, a] += bits[:, g]
            w[:, b] += 1 - bits[:, g]
        more = (w[:, None, :] > w[:, :, None]).sum(axis=2)              # other teams with strictly more wins than team t
        geq = (w[:, None, :] >= w[:, :, None]).sum(axis=2) - 1          # other teams with at least as many wins
        lo_more, hi_geq, lo_geq = np.minimum(lo_more, more.min(axis=0)), np.maximum(hi_geq, geq.max(axis=0)), np.minimum(lo_geq, geq.min(axis=0))
    return ["clinched" if hi_geq[t] <= k - 1 else "eliminated" if lo_more[t] >= k else "alive (points tiebreak only)" if lo_geq[t] >= k else "alive" for t in range(T)]
