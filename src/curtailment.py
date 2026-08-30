"""Deriving curtailed volume from physical notifications and acceptances.

Curtailment is not published anywhere, so it has to be constructed. The method
I am following is to rebuild two minute by minute profiles for each unit day, 
the level the operator intended to generate (PN) and the level NESO instructed 
(BOAL) and then integrate the gap between them.

NESO issues a new instruction roughly every 20 minutes. Each new instruction replaces 
the previous one, extends the hold period and cancels the planned increase in output. 
If the data is read row by row, it shows increases that never happened and 
underestimates curtailment. Therefore, for each minute, I use the most recently 
accepted instruction that was active at that time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_HOUR = 60


def minute_grid(day: str) -> pd.DatetimeIndex:
    """Building the UTC minute grid for one local settlement day.

    Elexon's settlementDate is a local calendar date, so during BST the day
    actually starts at 23:00 UTC the day before. Building the grid on day
    treated as a UTC calendar date silently drops the last hour of BST days,
    since rows near a local midnight carry UTC timeFrom values that fall
    before a UTC based grid for that settlementDate would start.
    """
    start = pd.Timestamp(f"{day} 00:00", tz="Europe/London").tz_convert("UTC")
    end = start + pd.Timedelta(hours=23, minutes=59)
    return pd.date_range(start, end, freq="1min", tz="UTC")


def to_utc(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Forcing timestamp columns to UTC aware, since DuckDB returns them naive.

    Naming the format explicitly stops pandas falling back to parsing each
    value one at a time, which was the slow path on the full backfill.
    """
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_datetime(
            out[column], utc=True, format="ISO8601", errors="coerce"
        )
    return out


def interpolate_levels(rows: pd.DataFrame, day: str) -> pd.Series:
    """Interpolating a level profile minute by minute, later rows winning.

    Used for PN, where rows do not supersede each other but can be restated.
    """
    frame = to_utc(rows, ("timeFrom", "timeTo")).sort_values("timeFrom")
    grid = minute_grid(day)
    level = pd.Series(np.nan, index=grid)

    for row in frame.itertuples():
        window = grid[(grid >= row.timeFrom) & (grid < row.timeTo)]
        if window.empty:
            continue
        span = (row.timeTo - row.timeFrom).total_seconds()
        elapsed = (window - row.timeFrom).total_seconds()
        level.loc[window] = row.levelFrom + (row.levelTo - row.levelFrom) * elapsed / span

    return level


def effective_instruction(rows: pd.DataFrame, day: str) -> pd.Series:
    """Rebuilding the instructed level, respecting superseded acceptances.

    Later acceptances override earlier ones for any minute they both cover,
    which is what makes a rolling hold at zero visible instead of a sequence
    of ramps that were cancelled before they happened.
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
        elapsed = (window - row.timeFrom).total_seconds()
        values = row.levelFrom + (row.levelTo - row.levelFrom) * elapsed / span
        newer = issued[window].isna() | (issued[window] <= row.acceptanceTime)
        level.loc[window[newer]] = np.asarray(values)[newer.to_numpy()]
        issued.loc[window[newer]] = row.acceptanceTime

    return level


def curtailed_profile(pn_rows: pd.DataFrame, boal_rows: pd.DataFrame, day: str) -> pd.Series:
    """Returning curtailed MW minute by minute for one unit-day.

    Where there is no instruction the unit follows its own notification, so
    filling the instructed level with PN makes those minutes zero rather than
    missing.
    """
    pn = interpolate_levels(pn_rows, day)
    boal = effective_instruction(boal_rows, day) if not boal_rows.empty else pd.Series(
        np.nan, index=minute_grid(day)
    )
    return (pn - boal.fillna(pn)).clip(lower=0)


def to_settlement_periods(profile: pd.Series, day: str) -> pd.DataFrame:
    """Aggregating a minute profile into settlement period volumes.

    Settlement periods run from local midnight rather than UTC midnight, so I
    convert to London time before numbering them. Without this the half hours
    either side of a UTC day boundary get numbered from two different daily
    runs and the same minute is counted twice.
    """
    frame = profile.rename("curtailed_mw").to_frame()
    local = frame.index.tz_convert("Europe/London")

    frame["settlementDate"] = local.date.astype(str)
    minutes_in = local.hour * 60 + local.minute
    frame["settlementPeriod"] = (minutes_in // 30) + 1

    grouped = frame.groupby(["settlementDate", "settlementPeriod"], as_index=False).agg(
        curtailed_mw_mean=("curtailed_mw", "mean"),
        minutes=("curtailed_mw", "size"),
    )
    grouped["curtailed_mwh"] = (
        grouped["curtailed_mw_mean"] * grouped["minutes"] / MINUTES_PER_HOUR
    )
    return grouped
