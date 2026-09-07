"""Building the curtailment cost feature from BOD bid prices.

For each unit and settlement period I take the most negative bid price
(pairId = -1, the deepest curtailment offer) and join it to the curtailment
target. The cost for that unit-period is curtailed_mwh x abs(bid_price).
I then sum to national level and join to the feature table.

Run:  python scripts/15_build_cost_feature.py
"""

import glob
import pandas as pd
from src import config


def load_bod() -> pd.DataFrame:
    files = glob.glob(str(config.RAW / "BOD" / "*" / "*.parquet"))
    print(f"Loading {len(files):,} BOD files...")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        frames.append(df)
    bod = pd.concat(frames, ignore_index=True)
    print(f"Total BOD rows: {len(bod):,}")
    bod = bod[bod["pairId"] == -1].copy()
    bod["settlementDate"] = bod["settlementDate"].astype(str)
    bod["settlementPeriod"] = bod["settlementPeriod"].astype(int)
    bod["bid"] = pd.to_numeric(bod["bid"], errors="coerce")
    bod = (
        bod.groupby(["bmUnit", "settlementDate", "settlementPeriod"])["bid"]
        .min()
        .reset_index()
        .rename(columns={"bid": "bid_price_gbp_mwh"})
    )
    return bod


def main() -> None:
    target = pd.read_parquet(config.PROCESSED / "curtailment_by_unit_period.parquet")
    print(f"Target rows: {len(target):,}")
    bod = load_bod()
    print(f"BOD unit-periods after filter: {len(bod):,}")
    merged = target.merge(
        bod,
        on=["bmUnit", "settlementDate", "settlementPeriod"],
        how="left",
    )
    merged["curtailment_cost_gbp"] = (
        merged["curtailed_mwh"] * merged["bid_price_gbp_mwh"].abs()
    )
    match_rate = merged["bid_price_gbp_mwh"].notna().mean()
    print(f"Match rate: {match_rate:.1%}")
    print(f"Mean bid price: {merged['bid_price_gbp_mwh'].abs().mean():.1f} GBP/MWh")
    print(f"Total estimated cost: {merged['curtailment_cost_gbp'].sum() / 1e6:.0f}m GBP")
    national = merged.groupby(
        ["settlementDate", "settlementPeriod"], as_index=False
    ).agg(
        curtailed_mwh=("curtailed_mwh", "sum"),
        curtailment_cost_gbp=("curtailment_cost_gbp", "sum"),
        avg_bid_price_gbp_mwh=("bid_price_gbp_mwh", lambda x: x.abs().mean()),
    )
    out = config.PROCESSED / "curtailment_cost_by_period.parquet"
    national.to_parquet(out, index=False)
    print(f"Wrote {len(national):,} rows to {out}")
    print(f"3 year total cost: {national['curtailment_cost_gbp'].sum() / 1e6:.0f}m GBP")
    print(f"Annual average: {national['curtailment_cost_gbp'].sum() / 3 / 1e6:.0f}m GBP/year")


if __name__ == "__main__":
    main()
