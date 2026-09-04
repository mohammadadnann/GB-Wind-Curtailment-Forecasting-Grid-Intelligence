"""Build and save the half-hourly national feature table.

Run:  python scripts/07_build_features.py
"""

from src import config, features


def main() -> None:
    frame = features.build_feature_table()

    out = config.PROCESSED / "features.parquet"
    frame.to_parquet(out)

    print(f"Wrote {len(frame):,} rows x {len(frame.columns)} columns to {out}")
    print(f"\nDate range: {frame.index.min()} to {frame.index.max()}")
    print(f"\nMissing values per column (top 10):")
    missing = frame.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10).to_string())
    print(f"\nColumns: {list(frame.columns)}")


if __name__ == "__main__":
    main()