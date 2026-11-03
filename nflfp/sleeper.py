"""Injury status from Sleeper's public API (no account or key needed): https://docs.sleeper.com

`/players/nfl` is a ~15 MB dump of every player; Sleeper asks for at most one fetch a day, so it is cached
for 24 hours. The status is as of *now*, so it only makes sense for projecting the upcoming week.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.sleeper.app/v1/players/nfl"
CACHE = Path(".cache/sleeper_players.json")
TTL_HOURS = 24
OUT_STATUSES = ("Out", "IR")          # Sleeper `injury_status` values treated as unavailable
TEAM = {"LAR": "LA"}                  # Sleeper team code -> nflverse
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def load_players(refresh: bool = False) -> dict:
    if CACHE.exists() and not refresh and (time.time() - CACHE.stat().st_mtime) / 3600 < TTL_HOURS:
        return json.loads(CACHE.read_text())
    for attempt in range(5):  # the shared proxy resets connections now and then
        try:
            with urllib.request.urlopen(URL, timeout=120) as r:
                text = r.read().decode()
            break
        except (urllib.error.URLError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(text)
    return json.loads(text)


def norm(name: str) -> str:
    """Lower-case, no punctuation, no Jr./III style suffixes - for matching names across sources."""
    words = "".join(c for c in name.lower().replace(".", "") if c.isalpha() or c == " ").split()
    return " ".join(w for w in words if w not in SUFFIXES)


def unavailable(players: dict, known: pd.DataFrame, statuses: tuple[str, ...] = OUT_STATUSES) -> pd.DataFrame:
    """Players Sleeper lists with an injury status in `statuses`, mapped to nflverse ids.

    `known` has one row per nflverse player (columns player_id, player_display_name, team, position). Sleeper's
    `gsis_id` (= nflverse id) is empty for many players, so we fall back to name + team + position.
    Returns columns player_id, name, team, position, status; Sleeper players we cannot match are omitted
    (they have no usage in our data anyway).
    """
    ids = set(known["player_id"])
    by_key = {(norm(r.player_display_name), r.team, r.position): r.player_id for r in known.itertuples()}
    rows = []
    for pl in players.values():
        status = pl.get("injury_status")
        if status not in statuses:
            continue
        name = f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip()
        team = TEAM.get(pl.get("team"), pl.get("team"))
        pid = pl.get("gsis_id") if pl.get("gsis_id") in ids else by_key.get((norm(name), team, pl.get("position")))
        if pid is not None:
            rows.append(dict(player_id=pid, name=name, team=team, position=pl.get("position"), status=status))
    return pd.DataFrame(rows, columns=["player_id", "name", "team", "position", "status"]).drop_duplicates("player_id")
