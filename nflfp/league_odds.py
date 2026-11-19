"""Playoff and championship odds for a Sleeper league, taking Sleeper's own projections as the truth.

    uv run python -m nflfp.league_odds <sleeper username> [--league NAME] [--sims 200000] [--sd 23] [--include-reserve] [--include-taxi]

Everything comes from Sleeper's public API: standings, the remaining schedule, the playoff bracket, rosters, and the weekly projections feed
(`api.sleeper.app/projections/nfl/{season}/{week}`), which is turned into points under the league's own scoring settings (kickers and defenses included).
Each team starts its best projected legal lineup every remaining week; the week's score is Normal(that projection, sd). `sd` is measured from the league's own
finished weeks: actual team score minus the feed's projection for the players that team actually started. Rosters are today's: no waiver pickups or trades.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import season_odds as so
from . import sleeper

FEED = "https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=K&position[]=DEF"
CACHE = Path(".cache/sleeper_feed")
SLOT_POS = ("QB", "RB", "WR", "TE", "K", "DEF")
ALIAS = {"fgm_50p": "fgm_50_59", "fgmiss_50p": "fgmiss_50_59"}   # the feed lumps 50+ yard kicks; the league may score 50-59 and 60+ separately (50-59 used)
NO_STATS = {"adp_dd_ppr", "pos_adp_dd_ppr"}
# The feed's first-down fields are placeholders (rush_fd = rush_yd / 10, rec_fd = rec_yd / 10: more first downs than catches), and its defense return-yard fields
# (def_pr_yd, def_kr_yd) run ~8x and ~3x its own pr_yd. For leagues that score them we replace first downs by volume x the nflverse 2023-25 regular-season rate
# for the position, and drop the defense return yards (they, with 4th-down stops, forced punts and interception-return yards, go into the measured DEF constant).
FD_PER_CARRY = {"QB": 0.346, "RB": 0.224, "WR": 0.293, "TE": 0.401}
FD_PER_REC = {"RB": 0.324, "WR": 0.597, "TE": 0.511, "QB": 0.368}
FD_PER_PASS = 0.336
DEF_BAD = ("def_pr_yd", "def_kr_yd")


def points(stats: dict, scoring: dict, position: str) -> float | None:
    """League-scoring points of a feed stat line; None when the line holds nothing but draft-rank fields (bye week, no projection)."""
    s = {k: v for k, v in stats.items() if k not in NO_STATS}
    if not s:
        return None
    for src, dst in ALIAS.items():
        if src in s and dst not in s and scoring.get(dst):
            s[dst] = s.pop(src)
    if position == "DEF":
        s = {k: v for k, v in s.items() if k not in DEF_BAD}
    elif position in FD_PER_CARRY:
        s["rush_fd"] = FD_PER_CARRY[position] * s.get("rush_att", 0.0)
        s["rec_fd"] = FD_PER_REC[position] * s.get("rec", 0.0)
        s["pass_fd"] = FD_PER_PASS * s.get("pass_att", 0.0)
    return so.points_from_stats(s, scoring)


def feed(season: str, week: int, ttl_hours: float | None) -> list[dict]:
    """One week of Sleeper projections (cached: finished weeks forever, future weeks for `ttl_hours`)."""
    f = CACHE / f"{season}_{week}.json"
    if f.exists() and (ttl_hours is None or (time.time() - f.stat().st_mtime) < ttl_hours * 3600):
        return json.loads(f.read_text())
    for a in range(5):
        r = subprocess.run(["curl", "-sS", "-g", "-m", "120", FEED.format(season=season, week=week)], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            break
        except Exception:
            if a == 4:
                raise RuntimeError(f"projection feed failed for week {week}: {r.stderr[:200]}")
            time.sleep(2 ** (a + 1))
    d = [{"player_id": x["player_id"], "position": x["player"]["position"], "stats": x["stats"]} for x in d if x["player"]["position"] in SLOT_POS]
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d))
    return d


def week_points(season: str, week: int, scoring: dict, current: int, def_extra: float = 0.0) -> dict[str, tuple[str, float]]:
    """{player id: (position, projected points)} for the players the feed projects that week; `def_extra` is added to every defense."""
    out = {}
    for x in feed(season, week, None if week < current else 6):
        p = points(x["stats"], scoring, x["position"])
        if p is not None:
            out[x["player_id"]] = (x["position"], p + (def_extra if x["position"] == "DEF" else 0.0))
    return out


def pick_league(leagues: list[dict], which: str | None) -> dict:
    if which:
        hit = [l for l in leagues if which.lower() in l["name"].lower() or which == l["league_id"]]
        if not hit:
            raise SystemExit(f"no league matching {which!r}: " + ", ".join(l["name"] for l in leagues))
        return hit[0]
    if len(leagues) > 1:
        print("Several leagues; using the first (pass --league to choose): " + ", ".join(l["name"] for l in leagues))
    return leagues[0]


def calibrate(L: dict, scoring: dict, season: str, upto: int, current: int) -> dict:
    """What the league's finished weeks (1..upto) say about the feed under this league's scoring.

    For every starter actually used: actual points vs the feed's projection (as converted by `points`). `def_extra` is the mean shortfall of the defense
    projection (scoring the feed does not project at all); `sd` and `bias` are the sd and mean of actual team score minus the projection of its starters
    (with `def_extra` added to the defense). `by_pos` shows the remaining bias by position."""
    rows = []
    for w in range(1, upto + 1):
        proj = week_points(season, w, scoring, current)
        for m in sleeper.get(f"league/{L['league_id']}/matchups/{w}"):
            if not m.get("starters") or m.get("points") is None:
                continue
            for p, pts in zip(m["starters"], m["starters_points"]):
                if p != "0":
                    pos, pr = proj.get(p, (None, 0.0))
                    rows.append((w, m["roster_id"], pos, pts, pr))
    d = pd.DataFrame(rows, columns=["week", "roster", "pos", "actual", "proj"])
    def_extra = float((d.actual - d.proj)[d.pos == "DEF"].mean())
    d["proj"] += np.where(d.pos == "DEF", def_extra, 0.0)
    tw = d.groupby(["week", "roster"])[["actual", "proj"]].sum()
    resid = tw.actual - tw.proj
    by_pos = (d.assign(resid=d.actual - d.proj).groupby("pos", dropna=False).resid.agg(["count", "mean"]).round(2))
    return dict(sd=float(resid.std(ddof=1)), bias=float(resid.mean()), n=len(resid), def_extra=def_extra, by_pos=by_pos)


def build(L: dict, rosters: list[dict], users: list[dict], state: dict, include_reserve: bool, include_taxi: bool, scoring: dict, def_extra: float = 0.0):
    """Everything `simulate` needs plus labels, from Sleeper's league objects."""
    st = L["settings"]
    season, current = L["season"], max(int(state["week"]), int(st.get("last_scored_leg", 0)) + 1)
    last_reg = st["playoff_week_start"] - 1
    if current > last_reg:
        raise SystemExit(f"the regular season is over (week {current}, playoffs start week {st['playoff_week_start']}); this tool simulates from the regular season")
    slots = so.lineup_slots(L["roster_positions"])
    name = {u["user_id"]: u["metadata"].get("team_name") or u["display_name"] for u in users}
    rid = [r["roster_id"] for r in rosters]
    idx = {r: i for i, r in enumerate(rid)}
    label = [name.get(r["owner_id"], f"roster {r['roster_id']}") for r in rosters]
    wins = np.array([r["settings"]["wins"] + 0.5 * r["settings"]["ties"] for r in rosters], float)
    pf = np.array([r["settings"]["fpts"] + r["settings"].get("fpts_decimal", 0) / 100 for r in rosters], float)
    losses = np.array([r["settings"]["losses"] + 0.5 * r["settings"]["ties"] for r in rosters], float)
    schedule = {}
    for w in range(current, last_reg + 1):
        pairs: dict[int, list[int]] = {}
        for m in sleeper.get(f"league/{L['league_id']}/matchups/{w}"):
            if m.get("matchup_id") is not None:
                pairs.setdefault(m["matchup_id"], []).append(idx[m["roster_id"]])
        schedule[w] = [tuple(v) for v in pairs.values() if len(v) == 2]

    bracket = sleeper.get(f"league/{L['league_id']}/winners_bracket")
    n_playoff = st["playoff_teams"]
    seed_key = (lambda i: -pf[i]) if st.get("playoff_seed_type") == 1 else (lambda i: (-wins[i], -pf[i]))
    seeded = sorted(range(len(rid)), key=seed_key)
    seed_of_roster = {rid[i]: s + 1 for s, i in enumerate(seeded[:n_playoff])}
    first_round = {m[k] for m in bracket for k in ("t1", "t2") if m.get(k) is not None}
    note = ""
    try:
        if not first_round <= set(seed_of_roster):
            raise KeyError
        br = so.parse_bracket(bracket, seed_of_roster)
    except KeyError:
        br, note = so.standard_bracket(n_playoff), "Sleeper's bracket does not match the standings; used a standard seeded bracket"
    rounds = so.round_weeks(max(m["round"] for m in br), st["playoff_week_start"], st.get("playoff_round_type", 0))
    weeks = list(range(current, last_reg + 1)) + [w for ws in rounds.values() for w in ws]

    meta = sleeper.load_players()
    mean, lineups, projs = {}, {}, {}
    for w in weeks:
        proj = projs[w] = week_points(season, w, scoring, current, def_extra)
        mean[w] = np.zeros(len(rid))
        lineups[w] = {}
        for i, r in enumerate(rosters):
            owned = set(r["players"] or [])
            if not include_reserve:
                owned -= set(r.get("reserve") or [])
            if not include_taxi:
                owned -= set(r.get("taxi") or [])
            cand = []
            for p in owned:
                if p not in proj:
                    continue
                inj = (meta.get(p) or {}).get("injury_status")
                if inj in so.OUT_FOR_SEASON or (inj == "Out" and w == current):
                    continue
                cand.append((p, proj[p][0], proj[p][1]))
            total, starters = so.best_lineup(cand, slots)
            mean[w][i] = total
            lineups[w][i] = starters
    s = so.Season(ids=label, wins=wins, pf=pf, schedule=schedule, mean=mean, n_playoff=n_playoff, bracket=br, rounds=rounds, losses=losses)
    return s, dict(season=season, current=current, last_reg=last_reg, rounds=rounds, note=note, lineups=lineups, slots=slots, proj=projs, meta=meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("username")
    ap.add_argument("--league")
    ap.add_argument("--sims", type=int, default=200000)
    ap.add_argument("--sd", type=float, help="weekly team-score sd; default: measured from the league's finished weeks")
    ap.add_argument("--shrink", type=float, default=1.0, help="pull each team's weekly projection toward that week's league average (1 = projections taken as true; "
                    "in this league's finished weeks the fitted slope of actual on projected differences was ~0.6)")
    ap.add_argument("--lineups", action="store_true", help="print every team's projected starters for the current week")
    ap.add_argument("--include-reserve", action="store_true", help="count players in IR slots as available")
    ap.add_argument("--include-taxi", action="store_true", help="count taxi-squad players as available")
    a = ap.parse_args()

    state = sleeper.get("state/nfl")
    uid = sleeper.user_id(a.username)
    L = pick_league(sleeper.get(f"user/{uid}/leagues/nfl/{state['league_season']}"), a.league)
    L = sleeper.get(f"league/{L['league_id']}")
    scoring = L["scoring_settings"]
    rosters, users = sleeper.get(f"league/{L['league_id']}/rosters"), sleeper.get(f"league/{L['league_id']}/users")
    state_week = max(int(state["week"]), int(L["settings"].get("last_scored_leg", 0)) + 1)
    cal = calibrate(L, scoring, L["season"], state_week - 1, state_week)
    s, info = build(L, rosters, users, state, a.include_reserve, a.include_taxi, scoring, cal["def_extra"])
    sd = a.sd if a.sd is not None else cal["sd"]
    if a.shrink != 1.0:
        s.mean = {w: m.mean() + a.shrink * (m - m.mean()) for w, m in s.mean.items()}
    print(f"finished weeks 1-{state_week - 1}: actual team score minus the feed's projection of the players started (league scoring): mean {cal['bias']:+.1f}, "
          f"sd {cal['sd']:.1f} over {cal['n']} team-weeks; defense constant {cal['def_extra']:+.1f} added to every projected defense")
    print("  remaining bias by position (actual - projected, per starter):", ", ".join(f"{k} {v:+.2f}" for k, v in cal["by_pos"]["mean"].items()))
    print(f"weekly team-score sd used: {sd:.1f}" + (" (given)" if a.sd is not None else ""))
    if info["note"]:
        print("note:", info["note"])

    res = so.simulate(s, sd=sd, n=a.sims)
    res.insert(1, "record", [f"{int(w)}-{int(l)}" for w, l in zip(s.wins, s.losses)])
    res.insert(2, "pf", s.pf)
    mine = [r for r in rosters if r["owner_id"] == uid]
    me = s.ids[[r["roster_id"] for r in rosters].index(mine[0]["roster_id"])] if mine else None

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = res.sort_values(["p_champion", "p_playoffs"], ascending=False).copy()
    for c in show.columns:
        if c.startswith("p_"):
            show[c] = (100 * show[c]).round(1)
    show["exp_wins"] = show["exp_wins"].round(2)
    show["pf"] = show["pf"].round(1)
    print(f"\n{L['name']}: {len(s.ids)} teams, top {s.n_playoff} make the playoffs (seeded by "
          f"{'points for' if L['settings'].get('playoff_seed_type') == 1 else 'record, then points for'}); regular season ends week {info['last_reg']}; "
          f"playoff rounds {info['rounds']}; {args_note(a)}; shrink {a.shrink:g}  [{a.sims:,} simulations]")
    print("(percentages)\n")
    print(show.to_string(index=False))
    print(f"\nsums: playoffs {res.p_playoffs.sum():.2f}, champion {res.p_champion.sum():.2f}, finalists {res.p_final.sum():.2f}")

    wk = pd.DataFrame({w: np.round(m, 1) for w, m in s.mean.items()}, index=s.ids)
    wk["avg"] = wk.mean(axis=1).round(1)
    print("\nProjected best-lineup score by week (Sleeper projections under league scoring):\n")
    print(wk.to_string())
    print("\nRemaining regular-season matchups:")
    for w, pairs in s.schedule.items():
        print(f"  week {w}: " + "; ".join(f"{s.ids[x]} ({s.mean[w][x]:.0f}) vs {s.ids[y]} ({s.mean[w][y]:.0f})" for x, y in pairs))
    if a.lineups:
        w = info["current"]
        for i, t in enumerate(s.ids):
            print(f"\n{t} - best lineup week {w}: {s.mean[w][i]:.1f}")
            for p in info["lineups"][w][i]:
                m = info["meta"].get(p) or {}
                pos, pts = info["proj"][w][p]
                print(f"   {pos:3s} {(m.get('first_name', '') + ' ' + m.get('last_name', '')).strip() or p:26s} {m.get('team') or '':4s} {pts:5.1f}  {m.get('injury_status') or ''}")
    if me:
        row = res[res.team == me].iloc[0]
        print(f"\n{me}: playoffs {100 * row.p_playoffs:.1f}%, title {100 * row.p_champion:.1f}%")


def args_note(a) -> str:
    return "reserve/taxi " + ("included" if a.include_reserve and a.include_taxi else "excluded" if not (a.include_reserve or a.include_taxi) else "partly included")


if __name__ == "__main__":
    main()
