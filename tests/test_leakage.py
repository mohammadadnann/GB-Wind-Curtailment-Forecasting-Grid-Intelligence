"""Leakage tests for the feature table.

These tests verify that lag features genuinely represent past data,
not same-day or future information. Tests check source timestamps
directly rather than scanning column names for suspicious words.

Prediction contract: forecast for day D issued at 11:00 UTC on D-1.
The minimum safe lag is D-2, which is shift(96) on a half-hourly series.
"""

import pandas as pd
import pytest

from src import features


@pytest.fixture(scope="module")
def spine():
    return features.half_hourly_spine()


@pytest.fixture(scope="module")
def target(spine):
    return features.load_target(spine)


def test_lag_2d_source_is_two_days_before_index(target):
    """curtailment_lag_2d at time T must equal curtailed_mwh at T - 48 hours."""
    lags = features.lag_features(target)
    series = target["curtailed_mwh"]

    # Check 100 random non-null points
    valid = lags["curtailment_lag_2d"].dropna()
    sample = valid.sample(100, random_state=42)

    for t, lag_value in sample.items():
        source_time = t - pd.Timedelta(hours=48)
        if source_time in series.index:
            assert abs(lag_value - series[source_time]) < 1e-6, (
                f"lag_2d at {t} = {lag_value}, "
                f"but curtailed_mwh at {source_time} = {series[source_time]}"
            )


def test_lag_7d_source_is_seven_days_before_index(target):
    """curtailment_lag_7d at time T must equal curtailed_mwh at T - 168 hours."""
    lags = features.lag_features(target)
    series = target["curtailed_mwh"]

    valid = lags["curtailment_lag_7d"].dropna()
    sample = valid.sample(100, random_state=42)

    for t, lag_value in sample.items():
        source_time = t - pd.Timedelta(hours=168)
        if source_time in series.index:
            assert abs(lag_value - series[source_time]) < 1e-6, (
                f"lag_7d at {t} = {lag_value}, "
                f"but curtailed_mwh at {source_time} = {series[source_time]}"
            )


def test_lag_columns_are_null_at_start_of_series(target):
    """No lag value should exist before enough history is available."""
    lags = features.lag_features(target)
    # First 96 periods (48 hours) cannot have a D-2 lag
    assert lags["curtailment_lag_2d"].iloc[:96].isna().all()
    # First 336 periods (168 hours) cannot have a D-7 lag
    assert lags["curtailment_lag_7d"].iloc[:336].isna().all()


def test_calendar_features_derived_from_index(spine):
    """Calendar features must match values derivable from the index itself."""
    cal = features.calendar_features(spine)
    local = spine.tz_convert("Europe/London")

    pd.testing.assert_series_equal(
        cal["month"], pd.Series(local.month, index=spine, name="month")
    )
    pd.testing.assert_series_equal(
        cal["hour"], pd.Series(local.hour, index=spine, name="hour")
    )


def test_weather_features_are_hourly_forward_filled(spine):
    """Weather values should repeat across the two half-hours within each hour."""
    weather = features.weather_features(spine)
    # For any given hour, the value at :00 and :30 should be identical
    # (because hourly data is forward-filled to half-hourly)
    on_the_hour = weather[spine.minute == 0]["wind_speed_100m"]
    on_the_half = weather[spine.minute == 30]["wind_speed_100m"]

    # Align by hour
    on_the_hour.index = on_the_hour.index.floor("h")
    on_the_half.index = on_the_half.index.floor("h")

    common = on_the_hour.index.intersection(on_the_half.index)
    pd.testing.assert_series_equal(
        on_the_hour.loc[common].reset_index(drop=True),
        on_the_half.loc[common].reset_index(drop=True),
        check_names=False,
    )