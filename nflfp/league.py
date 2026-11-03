"""Sleeper league tool: your matchup, a suggested lineup, and available players who beat someone on your roster.

    uv run python -m nflfp.league <sleeper username> [--week 9] [--margin 0.20] [--vs lineup|roster]

Everything about the league (rosters, matchup, scoring settings) comes from Sleeper's public API; all projections
are ours (see pipeline.project) - Sleeper's own projections are not used. Kickers and defenses are not modeled.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import model, pipeline, sleeper

SKILL = ("QB", "RB", "WR", "TE")
FLEX_SLOTS = {"FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
              "SUPER_FLEX": {"QB", "RB", "WR", "TE"}}


def scoring_from_league(settings: dict) -> tuple[model.Scoring, list[str]]:
    """Our scoring model from a Sleeper league's scoring_settings (+ notes where the league differs from what we model)."""
    ss = settings
    notes = []
    if abs(ss.get("rush_yd", 0.1) - ss.get("rec_yd", 0.1)) > 1e-9:
        notes.append("league scores rushing and receiving yards differently; the model uses the receiving rate for both")
    if ss.get("bonus_rec_te") or ss.get("rec_te"):
        notes.append("league has TE-specific reception bonuses, which are not modeled")
    return model.Scoring(ppr=ss.get("rec", 0.0), yard=ss.get("rec_yd", 0.1), td=ss.get("rec_td", 6.0),
                         pass_yard=ss.get("pass_yd", 0.04), pass_td=ss.get("pass_td", 4.0)), notes


def best_lineup(players: pd.DataFrame, slots: list[str]) -> pd.DataFrame:
    """Highest-projected lineup for the QB/RB/WR/TE and flex `slots` (K, DEF, BN are ignored).

    `players` has columns player, position, pts. Fixed positions are filled first, then flex slots from what is left
    (narrower flex types before wider ones). Returns one row per filled slot.
    """
    pool = players.dropna(subset=["pts"]).sort_values("pts", ascending=False)
    taken, rows = set(), []
    order = [s for s in slots if s in SKILL] + sorted((s for s in slots if s in FLEX_SLOTS), key=lambda s: len(FLEX_SLOTS[s]))
    for slot in order:
        ok = {slot} if slot in SKILL else FLEX_SLOTS[slot]
        cand = pool[pool.position.isin(ok) & ~pool.player.isin(taken)]
        if len(cand):
            pick = cand.iloc[0]
            taken.add(pick.player)
            rows.append(dict(slot=slot, player=pick.player, position=pick.position, pts=pick.pts))
    return pd.DataFrame(rows, columns=["slot", "player", "position", "pts"])


def find_upgrades(avail: pd.DataFrame, mine: pd.DataFrame, margin: float = 0.20) -> pd.DataFrame:
    """Available players who project at least `margin` (20%) above at least one same-position player on my roster.

    Only my players with a positive projection count as someone to beat (a bye/out player with 0 is a free spot, not a
    comparison). For each qualifying player we report how many of mine he beats by the margin and the biggest swap:
    the lowest-projected player of mine that he beats. `avail` needs player, position, pts; `mine` the same.
    """
    rows = []
    for a in avail.itertuples():
        cand = mine[(mine.position == a.position) & (mine.pts > 0)]
        beaten = cand[a.pts >= (1 + margin) * cand.pts]
        if len(beaten):
            drop = beaten.sort_values("pts").iloc[0]
            rows.append({**a._asdict(), "beats": len(beaten), "of_mine": len(cand), "replaces": drop.player,
                         "replaces_pts": drop.pts, "gain": a.pts - drop.pts, "gain_pct": a.pts / drop.pts - 1})
    out = pd.DataFrame(rows)
    if out.empty:
        return out.drop(columns=["Index"], errors="ignore")
    return out.drop(columns=["Index"]).sort_values("pts", ascending=False).reset_index(drop=True)


def _pick_league(leagues: list[dict], which: str | None) -> dict:
    if not leagues:
        raise SystemExit("No Sleeper leagues found for that user/season")
    if which:
        m = [l for l in leagues if which == l["league_id"] or which.lower() in l["name"].lower()]
        if len(m) != 1:
            raise SystemExit(f"--league '{which}' matches {len(m)} leagues: " + ", ".join(l["name"] for l in leagues))
        return m[0]
    if len(leagues) > 1:
        raise SystemExit("Several leagues; pick one with --league: " + ", ".join(f'{l["name"]} ({l["league_id"]})' for l in leagues))
    return leagues[0]


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nflfp.league", description=__doc__.split("\n")[0])
    ap.add_argument("username", help="Sleeper username (public)")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, help="week (default: Sleeper's current week)")
    ap.add_argument("--league", help="league id or part of its name (needed if you are in several)")
    ap.add_argument("--margin", type=float, default=0.20, help="how much better than a player on your roster (default 0.20 = 20%%)")
    ap.add_argument("--vs", choices=["lineup", "roster"], default="lineup",
                    help="compare with the players in your suggested lineup (default), or with anyone on your roster, bench included")
    ap.add_argument("--positions", nargs="+", default=list(SKILL), choices=SKILL)
    ap.add_argument("--top", type=int, default=25, help="rows of upgrades to print")
    ap.add_argument("--decay", type=float, default=model.DEFAULT_DECAY,
                    help=f"recency weight for carry/target shares (default {model.DEFAULT_DECAY}; 1.0 = all games equal)")
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--no-wind", action="store_true")
    ap.add_argument("--no-injuries", action="store_true")
    a = ap.parse_args()
    pd.options.display.float_format = "{:.1f}".format
    pd.options.display.width = 200

    uid = sleeper.user_id(a.username)
    league = _pick_league(sleeper.get(f"user/{uid}/leagues/nfl/{a.season}"), a.league)
    lid = league["league_id"]
    week = a.week or sleeper.get("state/nfl")["week"]
    scoring, notes = scoring_from_league(league["scoring_settings"])
    print(f"{league['name']} ({league['total_rosters']} teams), week {week}; scoring from the league: "
          f"{scoring.ppr} PPR, {scoring.pass_yard}/pass yd, {scoring.pass_td}/pass TD, {scoring.td}/TD")

    res = pipeline.project(a.season, week, scoring, lines=not a.no_lines, wind=not a.no_wind, injuries=not a.no_injuries,
                           decay=a.decay)
    notes += res.notes
    players = sleeper.load_players()
    known = res.pw.sort_values("week").groupby("player_id")[["player_display_name", "team", "position"]].last().reset_index()
    rosters = sleeper.get(f"league/{lid}/rosters")
    users = {u["user_id"]: u["display_name"] for u in sleeper.get(f"league/{lid}/users")}
    matchups = sleeper.get(f"league/{lid}/matchups/{week}")

    proj = res.proj.set_index("player_id")
    all_sids = {sid for r in rosters for sid in (r["players"] or [])}
    sid2pid = sleeper.map_ids(players, known, all_sids)
    flagged = {}   # nflverse id -> Questionable / Doubtful (Out and IR are already out of the projections)
    resolve = sleeper._resolver(known)
    for pl in players.values():
        if pl.get("injury_status") in ("Questionable", "Doubtful"):
            pid = resolve(pl)
            if pid:
                flagged[pid] = pl["injury_status"]

    def roster_table(sids) -> pd.DataFrame:
        rows = []
        for sid in sids:
            pl = players.get(sid, {})
            pid = sid2pid.get(sid)
            name = f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip() or sid
            pos = pl.get("position") or "DEF"
            rows.append(dict(sid=sid, player=name if pos != "DEF" else f"{sid} D/ST", position=pos, team=pl.get("team"),
                             status=pl.get("injury_status") or "", pts=float(proj.exp_pts[pid]) if pid in proj.index else np.nan))
        return pd.DataFrame(rows)

    mine_r = next((r for r in rosters if r["owner_id"] == uid), None)
    if mine_r is None:
        raise SystemExit("You are not on a roster in that league")
    mm = next((m for m in matchups if m["roster_id"] == mine_r["roster_id"]), None)
    slots = league["roster_positions"]

    # ---- matchup + suggested lineup ----------------------------------------------------------------------------
    for label, roster in (("YOU", mine_r), ("OPPONENT", None)):
        if roster is None:
            opp_m = next((m for m in matchups if mm and m["matchup_id"] == mm["matchup_id"] and m["roster_id"] != mm["roster_id"]), None)
            if opp_m is None:
                break
            roster = next(r for r in rosters if r["roster_id"] == opp_m["roster_id"])
            m = opp_m
        else:
            m = mm
        tbl = roster_table(roster["players"])
        starters = [s for s in (m or {}).get("starters", []) if s in set(tbl.sid)]
        taxi, ir = set(roster.get("taxi") or []), set(roster.get("reserve") or [])
        tbl["lineup"] = np.select([tbl.sid.isin(starters), tbl.sid.isin(taxi), tbl.sid.isin(ir)], ["START", "taxi", "IR"], "bench")
        order = {"START": 0, "bench": 1, "taxi": 2, "IR": 3}
        shown = tbl.assign(_o=tbl.lineup.map(order)).sort_values(["_o", "position", "pts"], ascending=[True, True, False]).drop(columns=["sid", "_o"])
        modeled = tbl[(tbl.lineup == "START") & tbl.pts.notna()]
        print(f"\n== {label}: {users.get(roster['owner_id'])} (record {roster['settings']['wins']}-{roster['settings']['losses']})")
        print(shown.to_string(index=False, na_rep="-"))
        print(f"   projected total of {len(modeled)} modeled starters (K/DEF not modeled): {modeled.pts.sum():.1f}")
        if label == "YOU":
            mine = tbl
            # taxi-squad and IR players can't be started in Sleeper without being activated first
            startable = tbl[tbl.lineup.isin(["START", "bench"])]
            sug = best_lineup(startable[["player", "position", "pts"]], slots)
            lineup_cmp = sug[["player", "position", "pts"]]
            cur = tbl[tbl.lineup == "START"]
            cur_sk = cur[cur.position.isin(SKILL)].pts.fillna(0).sum()
            gain = sug.pts.sum() - cur_sk
            print(f"\n   suggested lineup (QB/RB/WR/TE/flex): {sug.pts.sum():.1f} vs your current {cur_sk:.1f}  ->  {gain:+.1f}")
            # taxi players can be activated, but not moved back, so they are shown as a separate, explicit choice
            with_taxi = best_lineup(tbl[tbl.lineup.isin(["START", "bench", "taxi"])][["player", "position", "pts"]], slots)
            up = with_taxi.pts.sum() - sug.pts.sum()
            used = sorted(set(with_taxi.player) & set(tbl[tbl.lineup == "taxi"].player))
            if used:
                print(f"   activating from taxi ({', '.join(used)}) would add {up:+.1f} to the suggested lineup - a one-way move (no going back to taxi)")
            else:
                print("   no taxi player would improve the suggested lineup")
            if gain > 0.05:
                out_p = sorted(set(cur[cur.position.isin(SKILL)].player) - set(sug.player))
                in_p = sorted(set(sug.player) - set(cur.player))
                print(f"   swap in: {', '.join(in_p)};  out: {', '.join(out_p)}")

    # ---- upgrades available on the wire -------------------------------------------------------------------------
    rostered = set(sid2pid.values())
    skill_unmapped = [players[s] for s in all_sids if s not in sid2pid and players.get(s, {}).get("position") in SKILL]
    avail = res.proj[res.proj.position.isin(a.positions) & ~res.proj.player_id.isin(rostered)].copy()
    avail["injury"] = avail.player_id.map(flagged).fillna("")
    avail = avail.rename(columns={"exp_pts": "pts"})[["player", "position", "team", "opp", "pts", "injury"]]
    base = lineup_cmp if a.vs == "lineup" else mine[["player", "position", "pts"]].dropna(subset=["pts"])
    ups = find_upgrades(avail, base[base.position.isin(a.positions)], a.margin)
    if len(ups):
        ups = ups.sort_values("gain", ascending=False).reset_index(drop=True)
    who = "a player in your suggested lineup" if a.vs == "lineup" else "a player on your roster (bench included)"
    print(f"\n== Available {'/'.join(a.positions)} projected at least {a.margin:.0%} above {who} at the same position "
          f"(for each, the lowest-projected one he beats; sorted by gain)")
    if ups.empty:
        print("   none")
    else:
        show = ups.head(a.top)[["player", "position", "team", "opp", "pts", "injury", "replaces", "replaces_pts", "gain", "gain_pct", "beats", "of_mine"]]
        show = show.assign(gain_pct=(show.gain_pct * 100).round(0).astype(int).astype(str) + "%")
        print(show.to_string(index=False))
        print(f"   {len(ups)} qualify; showing {min(len(ups), a.top)}")
    dead = mine[mine.position.isin(a.positions) & ~(mine.pts > 0)]
    if len(dead):
        print("   roster spots with no projection this week (bye, out, or no usage): " + ", ".join(f"{r.player} ({r.status or r.position})" for r in dead.itertuples()))
    if skill_unmapped:
        names = ", ".join(f'{x.get("first_name", "")} {x.get("last_name", "")}'.strip() for x in skill_unmapped[:8])
        print(f"   note: {len(skill_unmapped)} rostered QB/RB/WR/TE players not found in our 2026 stats (no games yet, or a name mismatch): {names}")
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
