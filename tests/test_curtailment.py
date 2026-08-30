"""Tests for the curtailment derivation.

Each case is calculated manually, so the expected result 
does not come from the code. The replacement case is the 
most important because simple implementations often 
get it wrong.
"""

import pandas as pd

from src import curtailment

DAY = "2024-01-24"


def acceptance(time_from, time_to, level_from, level_to, issued, number=1):
    """Building one acceptance row in the shape the API returns."""
    return {
        "timeFrom": f"{DAY}T{time_from}Z",
        "timeTo": f"{DAY}T{time_to}Z",
        "levelFrom": level_from,
        "levelTo": level_to,
        "acceptanceTime": f"{DAY}T{issued}Z",
        "acceptanceNumber": number,
    }


def notification(time_from, time_to, level_from, level_to):
    """Building one physical notification row."""
    return {
        "timeFrom": f"{DAY}T{time_from}Z",
        "timeTo": f"{DAY}T{time_to}Z",
        "levelFrom": level_from,
        "levelTo": level_to,
    }


def test_flat_notification_interpolates_flat():
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    profile = curtailment.interpolate_levels(pn, DAY)
    assert profile.loc[f"{DAY} 00:30:00+00:00"] == 200
    assert profile.iloc[:60].eq(200).all()


def test_ramp_interpolates_linearly():
    # Ramping 0 to 60 MW over an hour means 30 MW at the halfway point
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 0, 60)])
    profile = curtailment.interpolate_levels(pn, DAY)
    assert profile.loc[f"{DAY} 00:30:00+00:00"] == 30


def test_later_acceptance_supersedes_earlier():
    # The first acceptance says ramp back up at 00:20, the second, issued
    # later, says stay at zero until 00:40. The later one must win.
    boal = pd.DataFrame([
        acceptance("00:00:00", "00:20:00", 0, 0, issued="23:00:00", number=1),
        acceptance("00:20:00", "00:40:00", 0, 100, issued="23:10:00", number=2),
        acceptance("00:20:00", "00:40:00", 0, 0, issued="23:30:00", number=3),
    ])
    profile = curtailment.effective_instruction(boal, DAY)
    assert profile.loc[f"{DAY} 00:30:00+00:00"] == 0


def test_uninstructed_minutes_are_not_curtailed():
    # No acceptance covers the period, so the unit follows its own
    # notification and nothing is curtailed
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    boal = pd.DataFrame(columns=["timeFrom", "timeTo", "levelFrom",
                                 "levelTo", "acceptanceTime", "acceptanceNumber"])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    assert profile.iloc[:60].eq(0).all()


def test_full_hold_curtails_the_whole_notification():
    # 200 MW held at zero for one hour is 200 MWh
    pn = pd.DataFrame([notification("00:00:00", "01:00:00", 200, 200)])
    boal = pd.DataFrame([acceptance("00:00:00", "01:00:00", 0, 0, issued="23:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    assert round(profile.iloc[:60].sum() / 60, 1) == 200.0


def test_settlement_periods_sum_to_the_profile():
    pn = pd.DataFrame([notification("00:00:00", "02:00:00", 200, 200)])
    boal = pd.DataFrame([acceptance("00:00:00", "02:00:00", 0, 0, issued="23:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    periods = curtailment.to_settlement_periods(profile, DAY)

    # Four half hour periods at 200 MW is 400 MWh in total
    assert round(periods["curtailed_mwh"].head(4).sum(), 1) == 400.0
    assert periods["settlementPeriod"].max() == 48


def test_periods_are_numbered_from_local_midnight():
    # A minute grid built on a UTC day boundary must map into the local
    # settlement day, otherwise the same half hour is counted from two runs
    pn = pd.DataFrame([notification("22:00:00", "23:59:00", 100, 100)])
    boal = pd.DataFrame([acceptance("22:00:00", "23:59:00", 0, 0, issued="21:00:00")])
    profile = curtailment.curtailed_profile(pn, boal, DAY)
    periods = curtailment.to_settlement_periods(profile, DAY)

    # Everything stays on one settlement date rather than spilling into the next
    assert periods["settlementDate"].nunique() == 1

    # January is GMT so local time equals UTC, and 22:00 to 23:59 is periods
    # 45 through 48
    curtailed = periods[periods["curtailed_mwh"] > 0]
    assert curtailed["settlementPeriod"].min() == 45
    assert curtailed["settlementPeriod"].max() == 48

def test_bst_day_grid_starts_at_2300_utc_the_day_before():
    # 1 July 2023 is BST, so local midnight is 23:00 UTC on 30 June. A UTC
    # calendar grid for this date would start three hours later and silently
    # drop the first three hours of the local settlement day.
    grid = curtailment.minute_grid("2023-07-01")
    assert grid[0] == pd.Timestamp("2023-06-30 23:00", tz="UTC")
    assert grid[-1] == pd.Timestamp("2023-07-01 22:59", tz="UTC")


def test_gmt_day_grid_starts_at_utc_midnight():
    # January is GMT, so local midnight equals UTC midnight and the grid
    # should be unaffected by the BST fix
    grid = curtailment.minute_grid(DAY)
    assert grid[0] == pd.Timestamp(f"{DAY} 00:00", tz="UTC")