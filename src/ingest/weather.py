"""Fetching forecast vintage weather from Open Meteo.

A forecast for day D is made at 11:00 on the previous day, so the model 
must use the weather forecast available at that time, not the actual 
weather recorded later. Open-Meteo’s historical forecast archive provides 
these past forecasts, which is why I use it instead of ERA5 reanalysis.

Using actual weather data would create data leakage and make the model appear more accurate than it would be in real use.
"""

from __future__ import annotations

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Hub height wind is what drives turbine output, and the power curve is roughly
# cubic in wind speed, so 100m speed matters more than anything else here
HOURLY_VARIABLES = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "surface_pressure",
]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def fetch_site(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """Fetching hourly forecast vintage weather for one point."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        response = client.get(BASE_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def fetch_clusters(start: str, end: str) -> pd.DataFrame:
    """Fetching weather for every cluster centre and stacking the results."""
    centres = pd.read_csv(config.REFERENCE / "weather_clusters.csv")

    frames = []
    for row in centres.itertuples():
        label = getattr(row, "name", None) or row.name_
        print(f"fetching {label} ({row.latitude:.2f}, {row.longitude:.2f})")
        site = fetch_site(row.latitude, row.longitude, start, end)
        site["cluster_name"] = label
        frames.append(site)

    return pd.concat(frames, ignore_index=True)