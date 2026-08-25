"""Client for the Elexon Insights API.

The API is public and needs no key. I keep every call going through one place
so retries, timeouts and the response shape are handled consistently.
"""

from __future__ import annotations

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src import config

TIMEOUT = httpx.Timeout(90.0, connect=15.0)
HEADERS = {"Accept": "application/json"}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def get_json(path: str, params: dict | list | None = None) -> dict | list:
    """Calling one Insights endpoint and returning the decoded body."""
    url = f"{config.ELEXON_BASE_URL}{path}"
    with httpx.Client(timeout=TIMEOUT, headers=HEADERS) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def to_frame(payload: dict | list) -> pd.DataFrame:
    """Normalising the two response shapes the API uses into a dataframe."""
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("results", []))
    else:
        rows = payload
    return pd.DataFrame(rows)


def fetch_bmunits() -> pd.DataFrame:
    """Fetching the full BM unit registration list."""
    return to_frame(get_json("/reference/bmunits/all"))