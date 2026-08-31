"""Tests for curtailment derivation.

Hand-computed expected values so the tests do not simply mirror the
implementation. The superseding-acceptance test is the most important:
a naive implementation gets it wrong by assuming ramps always execute.
"""

import pandas as pd
import pytest

from src import curtailment

DAY = "2024-01-24"


def acceptance(time_from, time_to, level_from, level_to, issued, number=1):
    return {
        "timeFrom": f"{DAY}T{time_from}Z",
        "timeTo": f"{DAY}T{time_to}Z",
        "levelFrom": level_from,
        "levelTo": level_to,
        "acceptanceTime": f"{DAY}T{issued}Z",
        "acceptanceNumber": number,
    }


def notification(time_from, time_to, level_from, level_to):
    return {
        "timeFrom": f"{DAY}T{time_from}Z",
        "timeTo": f"{DAY}T{time_to}Z",
        "levelFrom": level_from,
        "levelTo": level_to,
    }


# --- minute_grid ---

def test_normal_day_grid_has_1440_minutes():
    grid = curtailment.minute_grid("2024-01-24")
    assert len(grid) == 1440


def test_spring_clock_change_grid_has_1380_minutes():
    # 2024-03-31: clocks go forward, local day is 23 hours
    grid = curtailment.minute_grid("2024-03-31")
    assert len(grid) == 1380


def test_autumn_clock_change_grid_has_1500_minutes():
    # 2024-10-27: clocks go back, local day is 25 hours
    grid = curtailment.minute_grid("2024-10-27")
    assert len(grid) == 1500


def test_gmt_day_grid_starts_at_utc_midnight():
    grid = curtailment.minute_grid(DAY)
    assert grid[0] == pd.Timestamp(f"{DAY} 00:00", tz="UTC")


def test_bst_day_grid_starts_at_2300_utc_the_day_before():
    grid = curtailment.minute_grid("2023-07-01")
    assert grid[0] == pd.Timestamp("2023-06-30 23:00", tz="UTC")


# --- settlement period numbering ---

def test_normal_day_has_48_sequential_periods():
    pn = pd.DataFrame([notification("00:00:00", "23:59:00", 100, 100)])
    boal = pd.DataFrame([acceptance("00:00:00", "23:59:00", 0, 0, issued="23:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    periods = curtailment.to_settlement_periods(profile, DAY)
    assert list(periods["settlementPeriod"]) == list(range(1, 49))


def test_spring_clock_change_has_46_sequential_periods():
    day = "2024-03-31"
    pn = pd.DataFrame([{
        "timeFrom": f"{day}T00:00Z",
        "timeTo": f"{day}T22:59Z",
        "levelFrom": 100, "levelTo": 100,
    }])
    boal = pd.DataFrame([{
        "timeFrom": f"{day}T00:00Z",
        "timeTo": f"{day}T22:59Z",
        "levelFrom": 0, "levelTo": 0,
        "acceptanceTime": f"{day}T00:00Z",
        "acceptanceNumber": 1,
    }])
    profile = curtailment.curtailed_profile(pn, boal, day)
    periods = curtailment.to_settlement_periods(profile, day)
    assert list(periods["settlementPeriod"]) == list(range(1, 47))


def test_autumn_clock_change_has_50_sequential_periods():
    day = "2024-10-27"
    pn = pd.DataFrame([{
        "timeFrom": f"2024-10-26T23:00Z",
        "timeTo": f"2024-10-28T00:59Z",
        "levelFrom": 100, "levelTo": 100,
    }])
    boal = pd.DataFrame([{
        "timeFrom": f"2024-10-26T23:00Z",
        "timeTo": f"2024-10-28T00:59Z",
        "levelFrom": 0, "levelTo": 0,
        "acceptanceTime": f"2024-10-26T22:00Z",
        "acceptanceNumber": 1,
    }])
    profile = curtailment.curtailed_profile(pn, boal, day)
    periods = curtailment.to_settlement_periods(profile, day)
    assert list(periods["settlementPeriod"]) == list(range(1, 51))


def test_no_duplicate_or_missing_periods_on_autumn_clock_change():
    day = "2024-10-27"
    grid = curtailment.minute_grid(day)
    profile = pd.Series(0.0, index=grid)
    periods = curtailment.to_settlement_periods(profile, day)
    assert periods["settlementPeriod"].nunique() == 50
    assert periods["settlementPeriod"].min() == 1
    assert periods["settlementPeriod"].max() == 50


def test_energy_aggregation_correct_on_autumn_clock_change():
    # 25-hour day, 100 MW curtailed throughout = 2500 MWh
    day = "2024-10-27"
    pn = pd.DataFrame([{
        "timeFrom": f"2024-10-26T23:00Z",
        "timeTo": f"2024-10-28T00:59Z",
        "levelFrom": 100, "levelTo": 100,
    }])
    boal = pd.DataFrame([{
        "timeFrom": f"2024-10-26T23:00Z",
        "timeTo": f"2024-10-28T00:59Z",
        "levelFrom": 0, "levelTo": 0,
        "acceptanceTime": f"2024-10-26T22:00Z",
        "acceptanceNumber": 1,
    }])
    profile = curtailment.curtailed_profile(pn, boal, day)
    periods = curtailment.to_settlement_periods(profile, day)
    assert abs(periods["curtailed_mwh"].sum() - 2500.0) < 1.0


# --- PN interpolation ---

def test_flat_notification_interpolates_flat():
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    profile = curtailment.interpolate_levels(pn, DAY)
    assert profile.iloc[:60].eq(200).all()


def test_ramp_interpolates_linearly():
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 0, 60)])
    profile = curtailment.interpolate_levels(pn, DAY)
    assert profile.loc[f"{DAY} 00:30:00+00:00"] == 30


# --- acceptance superseding ---

def test_later_acceptance_supersedes_earlier():
    boal = pd.DataFrame([
        acceptance("00:00:00", "00:20:00", 0, 0, issued="23:00:00", number=1),
        acceptance("00:20:00", "00:40:00", 0, 100, issued="23:10:00", number=2),
        acceptance("00:20:00", "00:40:00", 0, 0, issued="23:30:00", number=3),
    ])
    profile = curtailment.effective_instruction(boal, DAY)
    assert profile.loc[f"{DAY} 00:30:00+00:00"] == 0


# --- curtailed_profile ---

def test_uninstructed_minutes_are_not_curtailed():
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    boal = pd.DataFrame(columns=["timeFrom", "timeTo", "levelFrom",
                                  "levelTo", "acceptanceTime", "acceptanceNumber"])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    assert profile.iloc[:60].eq(0).all()


def test_full_hold_curtails_the_whole_notification():
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    boal = pd.DataFrame([acceptance("00:00:00", "01:00:00", 0, 0, issued="23:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    assert round(profile.iloc[:60].sum() / 60, 1) == 200.0


def test_settlement_periods_sum_to_the_profile():
    pn = pd.DataFrame([notification("00:00:00", "02:00:00", 200, 200)])
    boal = pd.DataFrame([acceptance("00:00:00", "02:00:00", 0, 0, issued="23:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    periods = curtailment.to_settlement_periods(profile, DAY)
    assert round(periods["curtailed_mwh"].head(4).sum(), 1) == 400.0
    assert periods["settlementPeriod"].max() == 48


def test_periods_are_numbered_from_local_midnight():
    pn = pd.DataFrame([notification("22:00:00", "23:59:00", 100, 100)])
    boal = pd.DataFrame([acceptance("22:00:00", "23:59:00", 0, 0, issued="21:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    periods = curtailment.to_settlement_periods(profile, DAY)
    assert periods["settlementDate"].nunique() == 1
    curtailed = periods[periods["curtailed_mwh"] > 0]
    assert curtailed["settlementPeriod"].min() == 45
    assert curtailed["settlementPeriod"].max() == 48