"""Game-time weather per stadium from Open-Meteo (free, no key).

The interesting output is `wind_out_mph`: the component of the wind vector
along the home-plate -> center-field axis.  Positive = blowing out (helps
carry), negative = blowing in.

Geometry: Open-Meteo's wind_direction_10m is meteorological — the bearing the
wind comes FROM.  The direction of air travel is that +180 deg.  Projecting
onto the park azimuth:

    wind_out = speed * cos(travel_bearing - park_azimuth)
"""

from __future__ import annotations

import datetime as dt
import logging
import math

import requests

log = logging.getLogger(__name__)


def game_weather(lat: float, lon: float, game_time_utc: str, azimuth_deg: float) -> dict | None:
    """Hourly forecast nearest to first pitch. Returns temp/wind/out-component."""
    when = dt.datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
    day = when.date().strftime("%Y-%m-%d")
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=dict(
                latitude=lat, longitude=lon,
                hourly="temperature_2m,wind_speed_10m,wind_direction_10m",
                temperature_unit="fahrenheit", wind_speed_unit="mph",
                timezone="UTC", start_date=day, end_date=day,
            ),
            timeout=30,
        )
        r.raise_for_status()
        hourly = r.json()["hourly"]
    except Exception:
        log.warning("open-meteo failed for (%s,%s)", lat, lon, exc_info=True)
        return None

    times = [dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc) for t in hourly["time"]]
    idx = min(range(len(times)), key=lambda i: abs((times[i] - when).total_seconds()))

    temp_f = hourly["temperature_2m"][idx]
    speed = hourly["wind_speed_10m"][idx]
    from_dir = hourly["wind_direction_10m"][idx]
    if temp_f is None or speed is None or from_dir is None:
        return None
    travel = (from_dir + 180.0) % 360.0
    wind_out = speed * math.cos(math.radians(travel - azimuth_deg))
    return {
        "temp_f": float(temp_f),
        "wind_mph": float(speed),
        "wind_from_deg": float(from_dir),
        "wind_out_mph": float(wind_out),
    }
