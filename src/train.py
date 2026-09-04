"""Baselines and hurdle model for curtailment point forecasting.

Model architecture: two-stage hurdle.
  Stage 1 — LightGBM classifier: is this period constrained?
  Stage 2 — LightGBM regressor: given constraint, how much MWh?

The regressor trains only on constrained periods to avoid the zero-inflation
problem. Predictions combine both stages: E[Y] = P(constrained) * E[Y | constrained].

The target is log1p-transformed before regression to compress the long tail
of high-curtailment events. Predictions are back-transformed with expm1.

Baselines come first. A model that cannot beat persistence or the physical
heuristic has not learned anything useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "curtailed_mwh"

FEATURE_COLUMNS = [
    "curtailment_lag_2d",
    "curtailment_lag_7d",
    "curtailment_roll_mean_7d",
    "constrained_roll_mean_7d",
    "month",
    "hour",
    "day_of_week",
    "season",
    "is_weekend",
    "overnight_trough_flag",
    "wind_speed_100m",
    "wind_speed_100m_cubed",
    "wind_gust_10m",
    "wind_direction_sin",
    "wind_direction_cos",
]


# --- metrics ---

def mae(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1, keys=["a", "p"]).dropna()
    return float((aligned["a"] - aligned["p"]).abs().mean())


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1, keys=["a", "p"]).dropna()
    return float(np.sqrt(((aligned["a"] - aligned["p"]) ** 2).mean()))


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """MAE skill score relative to a baseline. Positive means improvement."""
    return float(1.0 - model_mae / baseline_mae)


# --- baselines ---

def baseline_zero(test: pd.DataFrame) -> pd.Series:
    """Predict zero curtailment for every period."""
    return pd.Series(0.0, index=test.index)


def baseline_persistence(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Predict the same period two days ago (D-2)."""
    combined = pd.concat([train[[TARGET]], test[[TARGET]]]).sort_index()
    return combined[TARGET].shift(96).reindex(test.index).fillna(0.0)


def baseline_weekly(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Predict the same period seven days ago (D-7)."""
    combined = pd.concat([train[[TARGET]], test[[TARGET]]]).sort_index()
    return combined[TARGET].shift(336).reindex(test.index).fillna(0.0)


def baseline_physical(test: pd.DataFrame) -> pd.Series:
    """Estimate curtailment from wind speed using a cubic power curve.

    max(0, estimated_output - assumed_limit) is the simplest physically
    grounded estimate. This requires no training and serves as a check
    that the ML model learns something beyond basic physics.
    """
    fleet_capacity_mw = 12_000
    rated_speed = 12.0
    assumed_limit_mw = 4_000

    estimated_output = (
        fleet_capacity_mw
        * (test["wind_speed_100m"] / rated_speed).clip(upper=1.0) ** 3
    )
    excess = estimated_output - assumed_limit_mw
    return excess.clip(lower=0)


# --- hurdle model ---

def fit_hurdle(train: pd.DataFrame) -> tuple:
    """Fit the two-stage hurdle model.

    Classifier trained on all periods.
    Regressor trained only on constrained periods to avoid zero-inflation.
    """
    import lightgbm as lgb

    clean = train.dropna(subset=FEATURE_COLUMNS + [TARGET])
    X = clean[FEATURE_COLUMNS]
    y_class = clean["is_constrained"]

    classifier = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    classifier.fit(X, y_class)

    constrained = clean[clean["is_constrained"] == 1]
    regressor = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    regressor.fit(constrained[FEATURE_COLUMNS], np.log1p(constrained[TARGET]))

    return classifier, regressor


def predict_hurdle(
    classifier, regressor, test: pd.DataFrame
) -> pd.Series:
    """Combine classifier and regressor into a point forecast."""
    X = test[FEATURE_COLUMNS]
    prob = classifier.predict_proba(X)[:, 1]
    log_volume = regressor.predict(X)
    volume = np.expm1(log_volume)
    return pd.Series(prob * volume, index=test.index)