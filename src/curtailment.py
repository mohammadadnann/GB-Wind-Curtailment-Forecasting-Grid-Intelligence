"""Curtailment derivation from physical notifications and acceptances.

Curtailment is not a published dataset. It is derived by comparing what a
wind unit planned to generate (PN) with what NESO instructed it to generate
(BOAL), minute by minute, then integrating the gap over each settlement period.

The key difficulty is that NESO reissues instructions every ~20 minutes. Each
new acceptance supersedes earlier ones for the minutes it covers, cancelling
ramps that never actually happened. Reading rows naively would understate
curtailment badly. The fix is to process acceptances in issuance-time order
and let later ones overwrite earlier ones at each minute.

Settlement periods run from local midnight to local midnight, not UTC midnight.
On UK clock change days this means 46 periods (spring) or 50 periods (autumn).
The grid must be built in local time and converted to UTC to handle this
correctly. Clock hour based period numbering fails on the autumn change because
the same clock hour appears twice  sequential numbering from position in the
sorted minute index is the correct approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_HOUR = 60


def minute_grid(day: str) -> pd.DatetimeIndex:
    """Build the UTC minute grid for one local settlement day.

    Grid runs from local midnight to the next local midnight, converted to UTC.
    This gives 1,380 minutes on spring clock-change days, 1,440 on normal days,
    and 1,500 on autumn clock-change days.
    """
    start = pd.Timestamp(day, tz="Europe/London")
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq="1min", tz="Europe/London", inclusive="left").tz_convert("UTC")


def to_utc(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Parse timestamp columns to UTC-aware datetimes."""
    out = frame.copy()
    for col in columns:
        out[col] = pd.to_datetime(out[col], utc=True, format="ISO8601", errors="coerce")
    return out


def interpolate_levels(rows: pd.DataFrame, day: str) -> pd.Series:
    """Interpolate a piecewise-linear level profile onto the minute grid.

    Used for PN. Each row defines a ramp from levelFrom to levelTo between
    timeFrom and timeTo. Later rows overwrite earlier ones where they overlap.
    """
    frame = to_utc(rows, ("timeFrom", "timeTo")).sort_values("timeFrom")
    grid = minute_grid(day)
    level = pd.Series(np.nan, index=grid)

    for row in frame.itertuples():
        window = grid[(grid >= row.timeFrom) & (grid < row.timeTo)]
        if window.empty:
            continue
        span = (row.timeTo - row.timeFrom).total_seconds()
        if span <= 0:
            continue
        elapsed = (window - row.timeFrom).total_seconds()
        level.loc[window] = row.levelFrom + (row.levelTo - row.levelFrom) * elapsed / span

    return level


def effective_instruction(rows: pd.DataFrame, day: str) -> pd.Series:
    """Reconstruct the instructed level at each minute, respecting superseding.

    Acceptances are processed in ascending acceptanceTime order. Each one
    overwrites any earlier acceptance for the minutes it covers, producing the
    actual hold the unit was subject to rather than the sequence of ramps that
    were cancelled.
    """
    frame = to_utc(rows, ("timeFrom", "timeTo", "acceptanceTime"))
    frame = frame.sort_values("acceptanceTime")

    grid = minute_grid(day)
    level = pd.Series(np.nan, index=grid)
    issued = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns, UTC]")

    for row in frame.itertuples():
        window = grid[(grid >= row.timeFrom) & (grid < row.timeTo)]
        if window.empty:
            continue
        span = (row.timeTo - row.timeFrom).total_seconds()
        if span <= 0:
            continue
        elapsed = (window - row.timeFrom).total_seconds()
        values = row.levelFrom + (row.levelTo - row.levelFrom) * elapsed / span
        newer = issued[window].isna() | (issued[window] <= row.acceptanceTime)
        level.loc[window[newer]] = np.asarray(values)[newer.to_numpy()]
        issued.loc[window[newer]] = row.acceptanceTime

    return level


def curtailed_profile(pn_rows: pd.DataFrame, boal_rows: pd.DataFrame, day: str) -> pd.Series:
    """Return curtailed MW at each minute for one unit-day.

    Where no instruction covers a minute the unit follows its own PN,
    so those minutes contribute zero curtailment.
    """
    pn = interpolate_levels(pn_rows, day)
    if boal_rows.empty:
        boal = pd.Series(np.nan, index=minute_grid(day))
    else:
        boal = effective_instruction(boal_rows, day)
    return (pn - boal.fillna(pn)).clip(lower=0)


def to_settlement_periods(profile: pd.Series, day: str) -> pd.DataFrame:
    """Aggregate a minute-level profile into settlement period MWh.

    Periods are numbered sequentially from 1 based on position within the
    local day, not from clock hours. This handles the autumn clock change
    correctly: the repeated clock hour would produce duplicate period numbers
    if clock hours were used, but sequential numbering always produces
    1 through 46, 48 or 50 with no gaps or duplicates.
    """
    grid = minute_grid(day)
    local_dates = grid.tz_convert("Europe/London").date

    frame = profile.rename("curtailed_mw").to_frame()
    frame["settlementDate"] = [str(d) for d in local_dates]

    # Sequential position within the local day, one-indexed
    date_arr = np.array([str(d) for d in local_dates])
    period = np.zeros(len(frame), dtype=int)
    for date in np.unique(date_arr):
        mask = date_arr == date
        positions = np.where(mask)[0]
        period[positions] = np.arange(1, len(positions) + 1)

    frame["settlementPeriod"] = (period - 1) // 30 + 1

    grouped = frame.groupby(["settlementDate", "settlementPeriod"], as_index=False).agg(
        curtailed_mw_mean=("curtailed_mw", "mean"),
        minutes=("curtailed_mw", "size"),
    )
    grouped["curtailed_mwh"] = (
        grouped["curtailed_mw_mean"] * grouped["minutes"] / MINUTES_PER_HOUR
    )
    return grouped