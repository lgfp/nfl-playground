"""Game-time weather from Open-Meteo (free, no API key; non-commercial use).

History: archive-api.open-meteo.com (ERA5 reanalysis). Forecast: api.open-meteo.com (16 days).
Weather is taken at the home team's stadium for the 3 hours after kickoff.
Neutral-site games are skipped (the schedule lists them under the nominal home stadium).
"""
from __future__ import annotations

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


def _get(url: str, cache_name: str | None, refresh: bool = False) -> dict:
    path = CACHE / cache_name if cache_name else None
    if path and path.exists() and not refresh:
        return json.loads(path.read_text())
    for attempt in range(5):  # the connection through the proxy is occasionally reset: retry with backoff
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            break
        except (urllib.error.URLError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    if path:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return json.loads(text)


def hourly(team: str, start: str, end: str, forecast: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Hourly UTC weather at `team`'s stadium: temp_f, precip_mm (in the hour before), snow_cm, wind_mph."""
    lat, lon = STADIUMS[team]
    q = dict(latitude=lat, longitude=lon, hourly=VARS, timezone="UTC", temperature_unit="fahrenheit", wind_speed_unit="mph")
    if forecast:  # forecasts change, so never cache them
        d = _get(f"{FORECAST}?{urllib.parse.urlencode({**q, 'forecast_days': 16})}", None)
    else:
        d = _get(f"{ARCHIVE}?{urllib.parse.urlencode({**q, 'start_date': start, 'end_date': end})}",
                 f"{team}_{start}_{end}.json", refresh)
    h = d["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True), "temp_f": h["temperature_2m"],
                       "precip_mm": h["precipitation"], "snow_cm": h["snowfall"], "wind_mph": h["wind_speed_10m"]})
    return df.set_index("time")


def kickoff_utc(gameday: str, gametime: str) -> pd.Timestamp:
    """nflverse gametime is US Eastern."""
    return pd.Timestamp(f"{gameday} {gametime}", tz="America/New_York").tz_convert("UTC")


def game_weather(schedule: pd.DataFrame, hours: int = 3, refresh: bool = False) -> pd.DataFrame:
    """One row per non-neutral game: precip_mm and snow_cm summed, temp_f / wind_mph averaged over the
    `hours` after kickoff. Index by game_id."""
    g = schedule[schedule["location"] != "Neutral"]
    rows = []
    for team, tg in g.groupby("home_team"):
        # +1 day: an evening kickoff plus the window runs past midnight UTC
        end = (pd.Timestamp(tg["gameday"].max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        h = hourly(team, tg["gameday"].min(), end, refresh=refresh)
        for r in tg.itertuples():
            k = kickoff_utc(r.gameday, r.gametime)
            w = h[(h.index > k) & (h.index <= k + pd.Timedelta(hours=hours))]
            rows.append(dict(game_id=r.game_id, precip_mm=w.precip_mm.sum(), snow_cm=w.snow_cm.sum(),
                             temp_f=w.temp_f.mean(), wind_mph=w.wind_mph.mean()))
    return pd.DataFrame(rows).set_index("game_id")


def check_complete(w: pd.DataFrame) -> pd.DataFrame:
    """Raise if any game is missing weather (e.g. reanalysis lag for very recent games)."""
    if w.isna().any(axis=1).any():
        raise ValueError(f"missing weather for: {list(w.index[w.isna().any(axis=1)])}")
    return w
