"""Building the curtailment target table for the whole Scottish fleet.

This runs the derivation across every unit day that had at least one
acceptance. Days with no acceptances cannot show curtailment, so they are
filled in as zero at the aggregation stage rather than being processed.

Run:  python scripts/05_build_target.py
"""

import glob

import pandas as pd
import pyarrow.parquet as pq

from src import config, curtailment


def usable_files(folder: str) -> list[str]:
    """Listing parquet files that carry a schema.

    An empty API response writes a parquet file with no columns at all, which
    pandas reads back but other readers reject, so I skip those here.
    """
    paths = sorted(glob.glob(str(config.RAW / folder / "*" / "*.parquet")))
    return [p for p in paths if pq.read_schema(p).names]


def unit_month(path: str) -> tuple[str, str]:
    """Pulling the unit and month labels out of a cache path."""
    parts = path.split("/")
    return parts[-1].replace(".parquet", ""), parts[-2]


def main() -> None:
    boalf_files = usable_files("BOALF")
    print(f"Processing {len(boalf_files):,} active unit-months")

    results = []
    for index, boalf_path in enumerate(boalf_files, start=1):
        unit, month = unit_month(boalf_path)
        pn_path = config.RAW / "PN" / month / f"{unit}.parquet"
        if not pn_path.exists():
            continue

        boalf = pd.read_parquet(boalf_path)
        pn = pd.read_parquet(pn_path)
        if boalf.empty or pn.empty:
            continue

        # Working one settlement day at a time, since the profiles are built
        # on a daily minute grid
        for day in sorted(boalf["settlementDate"].astype(str).unique()):
            boal_day = boalf[boalf["settlementDate"].astype(str) == day]
            pn_day = pn[pn["settlementDate"].astype(str) == day]
            if pn_day.empty:
                continue

            profile = curtailment.curtailed_profile(pn_day, boal_day, day)
            periods = curtailment.to_settlement_periods(profile, day)
            periods["bmUnit"] = unit
            results.append(periods)

        if index % 200 == 0:
            print(f"{index:,}/{len(boalf_files):,} unit-months")

    target = pd.concat(results, ignore_index=True)

    # Month boundaries can cause the same unit-day to be processed from two
    # different monthly cache files, since the API's UTC month windows do not
    # align with local settlement days. Keeping the larger value per key is
    # defensive - it should be identical either way, but this guards against
    # silently doubling MWh if the two runs ever produced different figures.
    before = len(target)
    target = target.sort_values("curtailed_mwh", ascending=False).drop_duplicates(
        subset=["bmUnit", "settlementDate", "settlementPeriod"], keep="first"
    )
    print(f"Removed {before - len(target):,} duplicate unit-periods from month boundary overlap")

    target = target[target["curtailed_mwh"] > 0].reset_index(drop=True)

    out = config.PROCESSED / "curtailment_by_unit_period.parquet"
    target.to_parquet(out, index=False)

    print(f"\nWrote {len(target):,} curtailed unit-periods to {out}")
    print(f"Total curtailed energy: {target['curtailed_mwh'].sum() / 1_000_000:,.2f} TWh")
    print("\nBy financial year:")
    dates = pd.to_datetime(target["settlementDate"])
    fy = dates.dt.year.where(dates.dt.month >= 4, dates.dt.year - 1)
    print((target.groupby(fy)["curtailed_mwh"].sum() / 1_000_000).round(2).to_string())


if __name__ == "__main__":
    main()