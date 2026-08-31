"""Fetching forecast vintage weather for the eight wind farm clusters.


"""

from src import config
from src.ingest import weather


def main() -> None:
    frame = weather.fetch_clusters(config.DATE_START, config.DATE_END)

    out = config.RAW / "weather_clusters.parquet"
    frame.to_parquet(out, index=False)

    print(f"\nWrote {len(frame):,} rows to {out}")
    print(f"date range: {frame['time'].min()} to {frame['time'].max()}")
    print()
    print(frame.groupby("cluster_name")["wind_speed_100m"].describe().round(1).to_string())


if __name__ == "__main__":
    main()