"""Game-time weather from Open-Meteo (free, no API key; non-commercial use).

History: archive-api.open-meteo.com (ERA5 reanalysis, ~5 days behind).
Recent + upcoming: api.open-meteo.com (`past_days` + 16-day forecast).
Weather is taken at the home team's stadium for the 3 hours after kickoff. Neutral-site games are skipped
(the schedule lists them under the nominal home stadium).

The free tier limits by IP and the sandbox shares one, so requests are batched (all stadiums in ONE call),
retried with backoff, and cached.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(".cache/weather")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
VARS = "temperature_2m,precipitation,snowfall,wind_speed_10m"
FORECAST_TTL_HOURS = 3     # forecasts change; re-use a cached one for this long
RECENT_DAYS = 8            # games newer than this come from the forecast API (the archive lags ~5 days)

# home stadium (lat, lon), nflverse team codes
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2577), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SF": (37.4032, -121.9695),
    "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}


def _get(url: str, cache_name: str | None = None, ttl_hours: float | None = None, refresh: bool = False):
    """GET json with retries (connection resets and 429s are intermittent through the shared proxy)."""
    path = CACHE / cache_name if cache_name else None
    if path and path.exists() and not refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if ttl_hours is None or age_h < ttl_hours:
            return json.loads(path.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            break
        except (urllib.error.URLError, OSError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** (attempt + 1), 20))
    if path:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return json.loads(text)


def _frames(teams: list[str], payload) -> dict[str, pd.DataFrame]:
    """Open-Meteo returns a list for several locations, a dict for one."""
    locs = payload if isinstance(payload, list) else [payload]
    out = {}
    for team, loc in zip(teams, locs):
        h = loc["hourly"]
        out[team] = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True), "temp_f": h["temperature_2m"],
                                  "precip_mm": h["precipitation"], "snow_cm": h["snowfall"],
                                  "wind_mph": h["wind_speed_10m"]}).set_index("time")
    return out


def _tag(teams: list[str]) -> str:
    return hashlib.md5("|".join(teams).encode()).hexdigest()[:10]  # stable across runs (hash() is not)


def _params(teams: list[str]) -> dict:
    return dict(latitude=",".join(str(STADIUMS[t][0]) for t in teams), longitude=",".join(str(STADIUMS[t][1]) for t in teams),
                hourly=VARS, timezone="UTC", temperature_unit="fahrenheit", wind_speed_unit="mph")


def archive_hourly(teams: list[str], start: str, end: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Hourly UTC history for several stadiums in one request."""
    teams = sorted(teams)
    q = {**_params(teams), "start_date": start, "end_date": end}
    key = f"archive_{start}_{end}_{_tag(teams)}.json"
    return _frames(teams, _get(f"{ARCHIVE}?{urllib.parse.urlencode(q)}", key, refresh=refresh))


def forecast_hourly(teams: list[str], past_days: int = 14, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Hourly UTC for the last `past_days` plus the next 16 days, several stadiums in one request."""
    teams = sorted(teams)
    q = {**_params(teams), "past_days": past_days, "forecast_days": 16}
    key = f"forecast_{_tag(teams)}.json"
    return _frames(teams, _get(f"{FORECAST}?{urllib.parse.urlencode(q)}", key, ttl_hours=FORECAST_TTL_HOURS, refresh=refresh))


def kickoff_utc(gameday: str, gametime: str) -> pd.Timestamp:
    """nflverse gametime is US Eastern."""
    return pd.Timestamp(f"{gameday} {gametime}", tz="America/New_York").tz_convert("UTC")


def _window(h: pd.DataFrame, k: pd.Timestamp, hours: int) -> dict:
    w = h[(h.index > k) & (h.index <= k + pd.Timedelta(hours=hours))].dropna(how="all")
    if len(w) < hours:  # not (fully) available yet, e.g. beyond the forecast horizon
        return dict(precip_mm=float("nan"), snow_cm=float("nan"), temp_f=float("nan"), wind_mph=float("nan"))
    return dict(precip_mm=w.precip_mm.sum(), snow_cm=w.snow_cm.sum(), temp_f=w.temp_f.mean(), wind_mph=w.wind_mph.mean())


def game_weather(schedule: pd.DataFrame, hours: int = 3, refresh: bool = False, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per non-neutral game (index game_id): precip_mm / snow_cm summed, temp_f / wind_mph averaged
    over the `hours` after kickoff. NaN where the weather is not available (beyond the 16-day forecast).

    Games older than RECENT_DAYS use the archive; newer and upcoming games use the forecast API.
    """
    g = schedule[schedule["location"] != "Neutral"].copy()
    today = today or pd.Timestamp.now(tz="UTC")
    cutoff = (today - pd.Timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
    old, new = g[g["gameday"] < cutoff], g[g["gameday"] >= cutoff]
    rows = []
    if len(old):
        end = (pd.Timestamp(old["gameday"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # kickoff + window crosses midnight UTC
        hist = archive_hourly(list(old["home_team"].unique()), old["gameday"].min(), end, refresh)
        for r in old.itertuples():
            rows.append(dict(game_id=r.game_id, **_window(hist[r.home_team], kickoff_utc(r.gameday, r.gametime), hours)))
    if len(new):
        # the forecast request is only worth making if a game falls inside its 14 + 16 day span
        span_start, span_end = today - pd.Timedelta(days=14), today + pd.Timedelta(days=16)
        near = new[[span_start <= kickoff_utc(r.gameday, r.gametime) <= span_end for r in new.itertuples()]]
        fc = forecast_hourly(list(near["home_team"].unique()), refresh=refresh) if len(near) else {}
        for r in new.itertuples():
            h = fc.get(r.home_team)
            rows.append(dict(game_id=r.game_id, **(_window(h, kickoff_utc(r.gameday, r.gametime), hours) if h is not None
                                                     else dict(precip_mm=float("nan"), snow_cm=float("nan"), temp_f=float("nan"), wind_mph=float("nan")))))
    return pd.DataFrame(rows).set_index("game_id")


def check_complete(w: pd.DataFrame) -> pd.DataFrame:
    """Raise if any game is missing weather (e.g. reanalysis lag for very recent games)."""
    if w.isna().any(axis=1).any():
        raise ValueError(f"missing weather for: {list(w.index[w.isna().any(axis=1)])}")
    return w
