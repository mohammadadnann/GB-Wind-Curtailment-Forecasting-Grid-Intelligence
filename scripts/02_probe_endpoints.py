"""Probing the Balancing Mechanism endpoints before the three year backfill.

Checking that each endpoint works with the from/to range parameters, printing
the schema, and counting rows for one high curtailment winter month on a few
units. That gives a real sizing figure instead of my estimate and tells us
whether the raw layer needs aggregating at ingest.

Run:  python scripts/02_probe_endpoints.py
"""

import pandas as pd

from src import config
from src.ingest import elexon

# A windy January, chosen because curtailment peaks in winter and this is the
# heaviest case for row counts
MONTH_START = "2026-01-01"
MONTH_END = "2026-02-01"

# One large offshore unit, one large onshore unit and one small onshore unit,
# so the average per unit is not skewed by picking only big sites
SAMPLE_UNITS = ["T_SGRWO-1", "T_WHILW-1", "T_ANSUW-1"]


def fetch_one(dataset: str, unit: str) -> pd.DataFrame:
    """Fetching one dataset for one unit across the probe month.

    The endpoint rejects wide ranges, so this goes through the weekly chunker.
    """
    if dataset in ("PN", "MELS"):
        def call(start, end):
            return elexon.fetch_physical(dataset, unit, start, end)
    elif dataset == "BOALF":
        def call(start, end):
            return elexon.fetch_acceptances(unit, start, end)
    else:
        def call(start, end):
            return elexon.fetch_bid_offer(unit, start, end)

    return elexon.fetch_range_chunked(call, MONTH_START, MONTH_END)


def count_scottish_units() -> int:
    """Counting the units we will actually backfill."""
    lookup = pd.read_csv(config.BMU_LOOKUP_PATH)
    usable = lookup["match_status"].isin(["accepted", "override"])
    return int((lookup["north_of_b6"] & usable).sum())


def main() -> None:
    units = count_scottish_units()
    print(f"Scottish units to backfill: {units}")

    totals: dict[str, int] = {}
    for unit in SAMPLE_UNITS:
        print(f"\n===== {unit} =====")
        for dataset in ["PN", "MELS", "BOALF", "BOD"]:
            try:
                frame = fetch_one(dataset, unit)
            except Exception as error:  # noqa: BLE001
                print(f"{dataset}: FAILED {type(error).__name__}")
                continue

            totals[dataset] = totals.get(dataset, 0) + len(frame)
            print(f"{dataset}: {len(frame):,} rows | {list(frame.columns)}")

    print("\n=== Sizing ===")
    grand = 0
    for dataset, count in totals.items():
        per_unit_month = count / len(SAMPLE_UNITS)
        projected = per_unit_month * units * 36
        grand += projected
        print(f"{dataset:>6}: {per_unit_month:>8,.0f} rows/unit/month -> {projected:>12,.0f} over 3 years")

    print(f"\nCombined projection: {grand:,.0f} raw rows")
    print(f"Calls for the full backfill: {units * 156 * 4:,}")


if __name__ == "__main__":
    main()