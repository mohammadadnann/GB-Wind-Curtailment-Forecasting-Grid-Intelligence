"""Leakage safe feature table for day ahead curtailment forecasting.

Prediction contract: forecast for day D is issued at 11:00 UTC on D-1.
Every feature in this table must be available at that time.

Weather features use the Open-Meteo historical forecast API, which stitches
NWP model runs into a continuous archive. The API does not expose per-value
issuance timestamps. We assume the D-1 06:00 UTC model run is the source,
which would be available well before the 11:00 cut-off. This is an
approximation — see docs/feature_availability.csv for the full assessment.

Lag conventions:
  shift(96)  = D-2 (two full settlement days before D)
  shift(336) = D-7 (seven days before D)
The original project used shift(48) and called it D-2, which is incorrect
for a half-hourly series where one day = 48 periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def half_hourly_spine() -> pd.DatetimeIndex:
    """Half-hourly UTC index spanning the full training window."""
    return pd.date_range(
        config.DATE_START, config.DATE_END, freq="30min", tz="UTC", inclusive="left"
    )


def load_target(spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Load curtailment target and align to the half-hourly spine.

    Periods with no curtailment are filled with zero. The target uses local
    settlement dates and period numbers, so we convert to UTC timestamps
    before aligning, handling ambiguous BST/GMT transitions by dropping them
    (they are a negligible fraction of the three-year series).
    """
    target = pd.read_parquet(config.PROCESSED / "curtailment_by_unit_period.parquet")

    local = (
        pd.to_datetime(target["settlementDate"])
        + pd.to_timedelta((target["settlementPeriod"] - 1) * 30, unit="min")
    )
    target["dt"] = (
        local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    target = target.dropna(subset=["dt"])

    national = target.groupby("dt")["curtailed_mwh"].sum()
    frame = national.reindex(spine, fill_value=0.0).rename("curtailed_mwh").to_frame()
    frame["is_constrained"] = (frame["curtailed_mwh"] > 0).astype(int)
    return frame


def lag_features(target: pd.DataFrame) -> pd.DataFrame:
    """Curtailment lags and rolling means, all safe at the D-1 11:00 cut-off.

    shift(96) gives D-2 on a half-hourly series (96 periods = 48 hours).
    shift(336) gives D-7 (336 periods = 168 hours).
    The rolling mean uses shift(96) as its base so it never touches D-1 or D.
    """
    series = target["curtailed_mwh"]
    constrained = target["is_constrained"]

    frame = pd.DataFrame(index=target.index)
    frame["curtailment_lag_2d"] = series.shift(96)
    frame["curtailment_lag_7d"] = series.shift(336)
    frame["curtailment_roll_mean_7d"] = (
        series.shift(96).rolling(336, min_periods=48).mean()
    )
    frame["constrained_roll_mean_7d"] = (
        constrained.shift(96).rolling(336, min_periods=48).mean()
    )
    return frame


def calendar_features(spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features derived entirely from the forecast timestamp."""
    local = spine.tz_convert("Europe/London")
    frame = pd.DataFrame(index=spine)
    frame["month"] = local.month
    frame["hour"] = local.hour
    frame["day_of_week"] = local.dayofweek
    frame["season"] = (local.month % 12) // 3
    frame["is_weekend"] = (local.dayofweek >= 5).astype(int)
    frame["overnight_trough_flag"] = local.hour.isin(range(1, 6)).astype(int)
    return frame


def weather_features(spine: pd.DatetimeIndex) -> pd.DataFrame:
    """Capacity-weighted national wind features from cluster weather data.

    Weather is hourly so we forward-fill to half-hourly. Wind direction is
    averaged using circular mean (sin/cos components weighted by capacity)
    to avoid the discontinuity at 0/360 degrees that arithmetic averaging
    would introduce.

    See docs/feature_availability.csv: issuance time is approximated.
    These features carry moderate leakage risk and are labelled APPROXIMATE.
    """
    weather = pd.read_parquet(config.RAW / "weather_clusters.parquet")
    clusters = pd.read_csv(config.REFERENCE / "weather_clusters.csv")
    weights = clusters.set_index("name")["capacity_mw"]

    def pivot(variable: str) -> pd.DataFrame:
        return weather.pivot_table(index="time", columns="cluster_name", values=variable)

    speed = pivot("wind_speed_100m").reindex(weights.index, axis=1)
    gusts = pivot("wind_gusts_10m").reindex(weights.index, axis=1)
    direction = pivot("wind_direction_100m").reindex(weights.index, axis=1)

    national_speed = (speed * weights).sum(axis=1) / weights.sum()
    national_gusts = gusts.max(axis=1)

    # Circular mean: convert degrees to radians, weight by capacity
    direction_rad = np.deg2rad(direction)
    sin_mean = (np.sin(direction_rad) * weights).sum(axis=1) / weights.sum()
    cos_mean = (np.cos(direction_rad) * weights).sum(axis=1) / weights.sum()

    frame = pd.DataFrame(index=spine)
    frame["wind_speed_100m"] = national_speed.reindex(spine, method="ffill")
    frame["wind_speed_100m_cubed"] = frame["wind_speed_100m"] ** 3
    frame["wind_gust_10m"] = national_gusts.reindex(spine, method="ffill")
    frame["wind_direction_sin"] = sin_mean.reindex(spine, method="ffill")
    frame["wind_direction_cos"] = cos_mean.reindex(spine, method="ffill")
    return frame


def boundary_features(spine: pd.DatetimeIndex) -> pd.DataFrame:
    """B6, SCOTEX and NKILGRMO boundary limit features from the NESO constraint flows dataset.

    The day ahead boundary limit is published before 11:00 UTC on D-1 and
    is therefore valid at the cut-off. I derive an outage proxy by comparing
    the current limit to its 30 day rolling median. A limit sitting well
    below its normal level usually indicates planned outage work.

    I kept three boundary groups because they capture different parts of the
    Scottish transmission network. SCOTEX and NKILGRMO are less correlated
    with B6 than with each other, so all three add independent signal.
    """
    limits = pd.read_csv(config.RAW / "neso_constraint_limits.csv", low_memory=False)

    limits["dt"] = (
        pd.to_datetime(limits["Date (GMT/BST)"], format="ISO8601")
        .dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    limits = limits.dropna(subset=["dt"])
    limits = limits[limits["Limit (MW)"].between(1, 50_000)]

    frame = pd.DataFrame(index=spine)

    boundary_groups = {
        "b6": ["SHARN", "SSHARN", "SSHARN3"],
        "scotex": ["SCOTEX"],
        "nkilgrmo": ["NKILGRMO"],
    }

    for name, groups in boundary_groups.items():
        mask = limits["Constraint Group"].isin(groups)
        boundary = limits[mask].groupby("dt").agg(
            limit_mw=("Limit (MW)", "mean")
        )

        frame[f"{name}_limit_mw"] = boundary["limit_mw"].reindex(spine)
        frame[f"{name}_limit_vs_30d_median"] = (
            frame[f"{name}_limit_mw"]
            / frame[f"{name}_limit_mw"].rolling("30D", min_periods=48).median()
        )
        frame[f"{name}_outage_flag"] = (
            frame[f"{name}_limit_vs_30d_median"] < 0.9
        ).astype(int)

    return frame


def build_feature_table() -> pd.DataFrame:
    """Assemble the complete half-hourly national feature table."""
    spine = half_hourly_spine()
    target = load_target(spine)

    parts = [
        target,
        lag_features(target),
        calendar_features(spine),
        weather_features(spine),
    ]

    # Boundary features require the NESO constraint limits file.
    # If it has not been downloaded yet, skip and warn rather than error.
    limits_path = config.RAW / "neso_constraint_limits.csv"
    if limits_path.exists():
        parts.append(boundary_features(spine))
    else:
        print(
            f"WARNING: {limits_path} not found. "
            "Boundary features omitted. Download from the NESO data portal."
        )

def build_feature_table() -> pd.DataFrame:
    """Assembling the complete half-hourly national feature table."""
    spine = half_hourly_spine()
    target = load_target(spine)

    parts = [
        target,
        lag_features(target),
        calendar_features(spine),
        weather_features(spine),
    ]

    limits_path = config.RAW / "neso_constraint_limits.csv"
    if limits_path.exists():
        parts.append(boundary_features(spine))
    else:
        print(
            f"WARNING: {limits_path} not found. "
            "Boundary features omitted. Download from the NESO data portal."
        )

    frame = pd.concat(parts, axis=1)

    # Joining curtailment cost derived from BOD bid prices
    cost_path = config.PROCESSED / "curtailment_cost_by_period.parquet"
    if cost_path.exists():
        cost = pd.read_parquet(cost_path)
        local = (
            pd.to_datetime(cost["settlementDate"])
            + pd.to_timedelta((cost["settlementPeriod"] - 1) * 30, unit="min")
        )
        cost["dt"] = (
            local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        cost = cost.dropna(subset=["dt"]).set_index("dt")
        frame["curtailment_cost_gbp"] = cost["curtailment_cost_gbp"].reindex(spine).fillna(0.0)
        frame["avg_bid_price_gbp_mwh"] = cost["avg_bid_price_gbp_mwh"].reindex(spine)
    else:
        print("WARNING: curtailment_cost_by_period.parquet not found. Run scripts/15_build_cost_feature.py first.")

    return frame