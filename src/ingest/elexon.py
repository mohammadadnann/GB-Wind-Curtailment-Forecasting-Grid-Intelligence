"""Client for the Elexon Insights API.

The API is public and needs no key. I keep every call going through one place
so retries, timeouts and the response shape are handled consistently.

The per BMU balancing endpoints take one unit and a from/to datetime range,
so I fetch a month at a time per unit and cache each result. That keeps the
call count low and makes the backfill resumable, since a rerun skips any month
already written to disk.
"""

from __future__ import annotations

from pathlib import Path

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


def fetch_physical(dataset: str, bm_unit: str, start: str, end: str) -> pd.DataFrame:
    """Fetching PN, QPN, MILS or MELS for one unit over a date range.

    The endpoint requires all three of bmUnit, from and to, and takes a single
    unit only. Leaving any of them out returns a 404 rather than a 400.
    """
    params = [
        ("bmUnit", bm_unit),
        ("from", f"{start}T00:00Z"),
        ("to", f"{end}T00:00Z"),
        ("dataset", dataset),
    ]
    return to_frame(get_json("/balancing/physical", params=params))


def fetch_acceptances(bm_unit: str, start: str, end: str) -> pd.DataFrame:
    """Fetching bid offer acceptances (BOALF) for one unit over a date range."""
    params = [
        ("bmUnit", bm_unit),
        ("from", f"{start}T00:00Z"),
        ("to", f"{end}T00:00Z"),
    ]
    return to_frame(get_json("/balancing/acceptances", params=params))


def fetch_bid_offer(bm_unit: str, start: str, end: str) -> pd.DataFrame:
    """Fetching bid and offer prices (BOD) for one unit over a date range."""
    params = [
        ("bmUnit", bm_unit),
        ("from", f"{start}T00:00Z"),
        ("to", f"{end}T00:00Z"),
    ]
    return to_frame(get_json("/balancing/bid-offer", params=params))


def cache_path(dataset: str, month: str, bm_unit: str) -> Path:
    """Building the parquet path for one dataset, month and unit."""
    folder = config.RAW / dataset / month
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{bm_unit}.parquet"


def cached_fetch(dataset: str, month: str, bm_unit: str, fetcher) -> pd.DataFrame:
    """Returning a month of data from disk, or fetching and saving it first.

    This is what makes the three year backfill resumable. If the run dies
    overnight I restart it and everything already written is skipped.
    """
    path = cache_path(dataset, month, bm_unit)
    if path.exists():
        return pd.read_parquet(path)

    frame = fetcher()
    frame.to_parquet(path, index=False)
    return frame

def fetch_range_chunked(fetcher, start: str, end: str, days: int = 7) -> pd.DataFrame:
    """Splitting a long date range into chunks the endpoint will accept.

    The per BMU endpoints reject wide ranges, so I request a week at a time
    and stitch the pieces back together.
    """
    edges = pd.date_range(start, end, freq=f"{days}D").strftime("%Y-%m-%d").tolist()
    if edges[-1] != end:
        edges.append(end)

    pieces = [fetcher(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)