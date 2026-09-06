"""Backfilling physical notifications and bid offer prices where they matter.

The acceptance backfill showed that a third of unit-months have no acceptances
at all, and those periods can never show curtailment. So instead of fetching PN
and BOD for every unit-month, I read the cached BOALF files and fetch only the
unit-months that had at least one acceptance.
"""

import glob
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src import config
from src.ingest import elexon


def active_unit_months() -> list[tuple[str, str]]:
    """Listing the unit-months that had at least one acceptance.

    A file with no rows means the unit was never instructed that month, so
    there is nothing to explain and no reason to fetch its physical data.
    """
    jobs = []
    for path in sorted(glob.glob(str(config.RAW / "BOALF" / "*" / "*.parquet"))):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        parts = path.split("/")
        month = parts[-2]
        unit = parts[-1].replace(".parquet", "")
        jobs.append((unit, month))
    return jobs


def month_edges(month: str) -> tuple[str, str]:
    """Turning a YYYY-MM label into start and end dates."""
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_one(dataset: str, unit: str, month: str) -> int:
    """Fetching one dataset for one unit-month, weekly chunked and cached."""
    start, end = month_edges(month)

    if dataset == "PN":
        def call():
            return elexon.fetch_range_chunked(
                lambda s, e: elexon.fetch_physical("PN", unit, s, e),
                start, end, days=config.CHUNK_DAYS,
            )
    else:
        def call():
            return elexon.fetch_range_chunked(
                lambda s, e: elexon.fetch_bid_offer(unit, s, e),
                start, end, days=config.CHUNK_DAYS,
            )

    frame = elexon.cached_fetch(dataset, month, unit, call)
    return len(frame)


def main() -> None:
    active = active_unit_months()
    jobs = [(dataset, unit, month) for unit, month in active for dataset in ("PN", "BOD")]
    print(f"{len(active):,} active unit-months -> {len(jobs):,} fetches")

    done = 0
    rows = {"PN": 0, "BOD": 0}
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, *job): job for job in jobs}
        for future, job in futures.items():
            
            try:
                rows[job[0]] += future.result()
            except Exception as error:  # noqa: BLE001
                print(f"FAILED {job}: {type(error).__name__}")
            done += 1
            if done % 400 == 0:
                print(f"{done:,}/{len(jobs):,} | PN {rows['PN']:,} | BOD {rows['BOD']:,}")

    print(f"\nDone. PN {rows['PN']:,} rows, BOD {rows['BOD']:,} rows")


if __name__ == "__main__":
    main()