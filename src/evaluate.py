"""Rolling-origin backtesting framework.

Expanding-window folds: each fold trains on all data up to a quarter
boundary and tests on the following calendar quarter. This is the only
correct evaluation strategy for a forecasting problem — random splits
would allow the model to train on data from after the point it predicts.

Minimum training requirement is six quarters (~18 months) to ensure every
fold has seen at least one full winter and one full summer before testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_folds(index: pd.DatetimeIndex, min_train_quarters: int = 6) -> list[Fold]:
    """Build expanding-window folds, one calendar quarter per test set."""
    starts = pd.date_range(
        index.min().normalize(), index.max(), freq="QS", tz=index.tz
    ).tolist()

    folds = []
    for i, test_start in enumerate(starts[min_train_quarters:], start=1):
        idx = starts.index(test_start)
        test_end = (
            starts[idx + 1]
            if idx + 1 < len(starts)
            else index.max() + pd.Timedelta(minutes=30)
        )
        folds.append(Fold(
            number=i,
            train_start=index.min(),
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
        ))
    return folds


def split(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature table into train and test rows for one fold."""
    train = frame[(frame.index >= fold.train_start) & (frame.index < fold.train_end)]
    test = frame[(frame.index >= fold.test_start) & (frame.index < fold.test_end)]
    return train, test


def describe_folds(folds: list[Fold]) -> pd.DataFrame:
    """Summarise folds as a table for a quick sanity check."""
    return pd.DataFrame([
        {
            "fold": f.number,
            "train_start": f.train_start.date(),
            "train_end": f.train_end.date(),
            "test_start": f.test_start.date(),
            "test_end": f.test_end.date(),
            "train_periods": None,
            "test_periods": None,
        }
        for f in folds
    ])