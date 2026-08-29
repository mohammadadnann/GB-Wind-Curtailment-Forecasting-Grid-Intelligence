"""Backfilling bid offer acceptances for the Scottish wind fleet.

BOALF comes first because curtailment is heavily concentrated. The probe found
one unit with 3,733 acceptance rows in a month while another had none at all,
so fetching physical notifications for every unit-month would spend most of the
run collecting data for periods where nothing happened. Once this stage is done
we know exactly which unit-months need PN and BOD.

"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src import config
from src.ingest import elexon


def scottish_units() -> list[str]:
    """Reading the units we settled on in the boundary lookup."""
    lookup = pd.read_csv(config.BMU_LOOKUP_PATH)
    usable = lookup["match_status"].isin(["accepted", "override"])
    scottish = lookup[lookup["north_of_b6"] & usable]
    return scottish["elexonBmUnit"].tolist()


def months() -> list[tuple[str, str, str]]:
    """Listing the months to backfill as (label, start, end)."""
    edges = pd.date_range(config.DATE_START, config.DATE_END, freq="MS")
    out = []
    for start in edges:
        end = start + pd.offsets.MonthBegin(1)
        out.append((start.strftime("%Y-%m"), start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d")))
    return out


def fetch_month(unit: str, label: str, start: str, end: str) -> int:
    """Fetching one unit-month of acceptances, weekly chunked and cached."""
    def call():
        return elexon.fetch_range_chunked(
            lambda s, e: elexon.fetch_acceptances(unit, s, e),
            start, end, days=config.CHUNK_DAYS,
        )

    frame = elexon.cached_fetch("BOALF", label, unit, call)
    return len(frame)


def main() -> None:
    units = scottish_units()
    periods = months()
    jobs = [(u, label, s, e) for label, s, e in periods for u in units]
    print(f"{len(units)} units x {len(periods)} months = {len(jobs):,} unit-months")

    done = 0
    rows = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_month, *job): job for job in jobs}
        for future in futures:
            job = futures[future]
            try:
                rows += future.result()
            except Exception as error:  # noqa: BLE001
                print(f"FAILED {job[0]} {job[1]}: {type(error).__name__}")
            done += 1
            if done % 200 == 0:
                print(f"{done:,}/{len(jobs):,} unit-months, {rows:,} rows so far")

    print(f"\nDone. {rows:,} acceptance rows across {len(jobs):,} unit-months")


if __name__ == "__main__":
    main()