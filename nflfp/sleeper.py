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


API = "https://api.sleeper.app/v1"


def get(path: str):
    """GET a Sleeper endpoint (public, read-only) with retries; returns parsed json."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=60) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))


def user_id(username: str) -> str:
    u = get(f"user/{username}")
    if not u or "user_id" not in u:
        raise SystemExit(f"Sleeper user '{username}' not found")
    return u["user_id"]


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


def _resolver(known: pd.DataFrame):
    """sleeper player dict -> nflverse id (or None): Sleeper's gsis_id when we know it, else name + team + position."""
    ids = set(known["player_id"])
    by_key = {(norm(r.player_display_name), r.team, r.position): r.player_id for r in known.itertuples()}

    def resolve(pl: dict):
        name = f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip()
        team = TEAM.get(pl.get("team"), pl.get("team"))
        return pl.get("gsis_id") if pl.get("gsis_id") in ids else by_key.get((norm(name), team, pl.get("position")))

    return resolve


def map_ids(players: dict, known: pd.DataFrame, sleeper_ids) -> dict:
    """{sleeper id: nflverse id} for the Sleeper ids we can match."""
    resolve = _resolver(known)
    out = {sid: resolve(players[sid]) for sid in sleeper_ids if sid in players}
    return {k: v for k, v in out.items() if v is not None}


def unavailable(players: dict, known: pd.DataFrame, statuses: tuple[str, ...] = OUT_STATUSES) -> pd.DataFrame:
    """Players Sleeper lists with an injury status in `statuses`, mapped to nflverse ids.

    `known` has one row per nflverse player (columns player_id, player_display_name, team, position). Sleeper's
    `gsis_id` (= nflverse id) is empty for many players, so we fall back to name + team + position.
    Returns columns player_id, name, team, position, status; Sleeper players we cannot match are omitted
    (they have no usage in our data anyway).
    """
    resolve = _resolver(known)
    rows = []
    for pl in players.values():
        status = pl.get("injury_status")
        if status not in statuses:
            continue
        name = f'{pl.get("first_name", "")} {pl.get("last_name", "")}'.strip()
        team = TEAM.get(pl.get("team"), pl.get("team"))
        pid = resolve(pl)
        if pid is not None:
            rows.append(dict(player_id=pid, name=name, team=team, position=pl.get("position"), status=status))
    return pd.DataFrame(rows, columns=["player_id", "name", "team", "position", "status"]).drop_duplicates("player_id")
