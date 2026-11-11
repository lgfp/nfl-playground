"""ESPN's weekly projections WITH variances (fantasy API, no login): for every projected stat ESPN publishes a mean and a variance.

    GET lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3?view=kona_playercard
        header x-fantasy-filter: {"players": {"filterIds": {"value": [ids]}, "filterStatsForExternalIds": {"value": [202610, ...]},
                                              "filterStatsForSourceIds": {"value": [1]}, "filterStatsForSplitTypeIds": {"value": [1]}}}
    an entry with statSourceId 1 / statSplitTypeId 1 is one week's projection ('externalId' = season*100 + week): `stats` = means, `variance` = variances,
    both keyed by ESPN stat id.

Points under a league's scoring are a linear combination of the stats, so mean and variance follow directly. ESPN gives no covariances, so the variance of the total
assumes the stats are independent (a lower bound on the true variance: yards and touchdowns move together).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nflfp import sleeper
from nflfp.model import Scoring

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3?view={view}"
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
# ESPN stat ids: 3 pass yds, 4 pass TD, 23 rush att, 24 rush yds, 25 rush TD, 42 rec yds, 43 rec TD, 53 receptions, 58 targets
WEIGHTS = lambda sc: {"3": sc.pass_yard, "4": sc.pass_td, "24": sc.yard, "25": sc.td, "42": sc.yard, "43": sc.td, "53": sc.ppr}
CACHE = Path(".cache/espn")


def _get(view: str, flt: dict, season: int = 2026, tries: int = 4):
    for a in range(tries):
        r = subprocess.run(["curl", "-sS", "-m", "120", "-A", "Mozilla/5.0", "-H", "x-fantasy-filter: " + json.dumps(flt), BASE.format(season=season, view=view)],
                           capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(2 ** (a + 1))
    raise RuntimeError("ESPN request failed: " + r.stderr[:200])


def player_list(limit: int = 1500) -> pd.DataFrame:
    """ESPN players (QB/RB/WR/TE by popularity): espn_id, name, position, ESPN pro team id."""
    f = CACHE / "players.pkl"
    if f.exists() and (time.time() - f.stat().st_mtime) < 24 * 3600:
        return pd.read_pickle(f)
    d = _get("kona_player_info", {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}, "filterSlotIds": {"value": [0, 2, 4, 6]}, "limit": limit,
                                              "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}})["players"]
    p = pd.DataFrame([{"espn_id": x["id"], "name": x["player"]["fullName"], "position": POS.get(x["player"]["defaultPositionId"]), "pro_team": x["player"]["proTeamId"],
                       "injury": x["player"].get("injuryStatus"), "owned": x["player"]["ownership"].get("percentOwned")} for x in d])
    CACHE.mkdir(parents=True, exist_ok=True)
    p.to_pickle(f)
    return p


def projections(ids: list[int], periods: list[int], batch: int = 60) -> pd.DataFrame:
    """One row per (player, week) with ESPN's projected stat means ('m<id>') and variances ('v<id>'). `periods` are season*100 + week."""
    rows = []
    for i in range(0, len(ids), batch):
        flt = {"players": {"filterIds": {"value": ids[i:i + batch]}, "filterStatsForExternalIds": {"value": periods},
                           "filterStatsForSourceIds": {"value": [1]}, "filterStatsForSplitTypeIds": {"value": [1]}}}
        for x in _get("kona_playercard", flt)["players"]:
            for e in x["player"].get("stats", []):
                if e.get("statSourceId") == 1 and e.get("statSplitTypeId") == 1:
                    r = {"espn_id": x["id"], "season": e["seasonId"], "week": e["scoringPeriodId"]}
                    r.update({f"m{k}": v for k, v in e.get("stats", {}).items()})
                    r.update({f"v{k}": v for k, v in e.get("variance", {}).items()})
                    rows.append(r)
    return pd.DataFrame(rows)


def to_points(P: pd.DataFrame, sc: Scoring = Scoring()) -> pd.DataFrame:
    """Adds espn_mean (points), espn_sd (assuming independent stats) and espn_td_sd_share (share of that variance from touchdowns)."""
    P = P.copy()
    w = WEIGHTS(sc)
    mean, var, tdv = np.zeros(len(P)), np.zeros(len(P)), np.zeros(len(P))
    for k, c in w.items():
        m = P[f"m{k}"].fillna(0).to_numpy(float) if f"m{k}" in P else 0.0
        v = P[f"v{k}"].fillna(0).to_numpy(float) if f"v{k}" in P else 0.0
        mean += c * m
        var += c * c * v
        if k in ("4", "25", "43"):
            tdv += c * c * v
    P["espn_mean"], P["espn_sd"], P["espn_td_var_share"] = mean, np.sqrt(var), tdv / np.maximum(var, 1e-9)
    return P
