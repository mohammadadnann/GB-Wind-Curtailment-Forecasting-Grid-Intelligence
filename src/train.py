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

# --- quantile models ---

QUANTILES = [0.1, 0.5, 0.9]


def fit_quantile_models(train: pd.DataFrame) -> dict:
    """Fit one quantile regressor per quantile on constrained periods only.

    Trained on log1p-transformed target, same as the point regressor.
    Each quantile is independent — no constraint tying them together,
    which is why the raw band needs conformal widening after fitting.
    """
    import lightgbm as lgb

    clean = train.dropna(subset=FEATURE_COLUMNS + [TARGET])
    constrained = clean[clean["is_constrained"] == 1]
    X = constrained[FEATURE_COLUMNS]
    y = np.log1p(constrained[TARGET])

    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        models[q] = model
    return models


def predict_quantiles(
    classifier, quantile_models: dict, test: pd.DataFrame
) -> pd.DataFrame:
    """Combine classifier with quantile regressors.

    Uses a zero-inflated mixture: for quantile q, if q falls below the
    zero-mass probability (1 - p_constrained), the quantile is zero.
    Only when q exceeds the zero mass does the continuous distribution
    contribute. This is the statistically correct treatment for a
    distribution with a point mass at zero.
    """
    X = test[FEATURE_COLUMNS]
    prob = classifier.predict_proba(X)[:, 1]
    prob_zero = 1 - prob

    result = pd.DataFrame(index=test.index)
    for q, model in quantile_models.items():
        volume = np.expm1(model.predict(X))
        below_zero_mass = q <= prob_zero
        result[f"p{int(q * 100)}"] = np.where(below_zero_mass, 0.0, volume)
    return result


def fit_conformal_widening(
    classifier,
    quantile_models: dict,
    calib: pd.DataFrame,
    target_coverage: float = 0.8,
) -> float:
    """Find the smallest symmetric widening factor that achieves target coverage.

    Fits on a held-out calibration slice the quantile models have not seen.
    Searches over widening factors from 1.0 to 5.0 in steps of 0.05.
    Returns the first factor achieving the target coverage.
    """
    predicted = predict_quantiles(classifier, quantile_models, calib)
    actual = calib[TARGET]

    aligned = pd.concat(
        [actual, predicted[["p10", "p50", "p90"]]], axis=1
    ).dropna()

    for factor in np.arange(1.0, 5.01, 0.05):
        lower = (aligned["p50"] - (aligned["p50"] - aligned["p10"]) * factor).clip(lower=0)
        upper = aligned["p50"] + (aligned["p90"] - aligned["p50"]) * factor
        inside = (aligned[TARGET] >= lower) & (aligned[TARGET] <= upper)
        if inside.mean() >= target_coverage:
            return round(float(factor), 3)

    return 5.0


def apply_conformal_widening(predicted: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Apply the calibrated widening factor to a quantile prediction."""
    result = predicted.copy()
    result["p10"] = (predicted["p50"] - (predicted["p50"] - predicted["p10"]) * factor).clip(lower=0)
    result["p90"] = predicted["p50"] + (predicted["p90"] - predicted["p50"]) * factor
    return result


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    """Pinball loss — the correct scoring rule for a quantile forecast."""
    aligned = pd.concat([actual, predicted], axis=1, keys=["a", "p"]).dropna()
    diff = aligned["a"] - aligned["p"]
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Fraction of actual values falling inside the predicted interval."""
    aligned = pd.concat(
        [actual, lower, upper], axis=1, keys=["a", "lo", "hi"]
    ).dropna()
    inside = (aligned["a"] >= aligned["lo"]) & (aligned["a"] <= aligned["hi"])
    return float(inside.mean())


# --- cost model ---

def fit_cost_model(train: pd.DataFrame):
    """Fit a LightGBM regressor for constraint cost in pounds.

    Cost is defined as curtailed_mwh * avg_bid_price_gbp_mwh. This is an
    estimate, not a settled figure — real settlement depends on system
    pricing rules not modelled here. The cost column must exist in the
    feature table for this to run.

    Trained on constrained periods only, same reasoning as the volume regressor.
    """
    import lightgbm as lgb

    clean = train.dropna(subset=FEATURE_COLUMNS + ["curtailment_cost_gbp"])
    constrained = clean[clean["is_constrained"] == 1]

    if len(constrained) == 0:
        raise ValueError("No constrained periods with cost data in training set.")

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(constrained[FEATURE_COLUMNS], np.log1p(constrained["curtailment_cost_gbp"]))
    return model


def predict_cost(classifier, cost_model, test: pd.DataFrame) -> pd.Series:
    """Combine classifier with cost regressor."""
    X = test[FEATURE_COLUMNS]
    prob = classifier.predict_proba(X)[:, 1]
    log_cost = cost_model.predict(X)
    return pd.Series(np.expm1(log_cost) * prob, index=test.index)


# --- battery simulation ---

BATTERY_POWER_MW = 50
BATTERY_ENERGY_MWH = 200
ROUND_TRIP_EFFICIENCY = 0.9
NOMINAL_PRICE_GBP_MWH = 60.0


def simulate_battery(curtailed_mwh: pd.Series, charge_periods: pd.Series) -> float:
    """Simulate a battery charging during curtailment and returning captured value.

    charge_periods is a boolean series deciding which periods to charge during.
    Using an explicit period set rather than a threshold ensures every strategy
    gets an identical charging budget, making perfect foresight a genuine ceiling.
    """
    state_of_charge = 0.0
    value_gbp = 0.0

    for t, curtailed in curtailed_mwh.items():
        if charge_periods.get(t, False) and curtailed > 0:
            charge = min(
                BATTERY_POWER_MW * 0.5,
                BATTERY_ENERGY_MWH - state_of_charge,
                curtailed,
            )
            state_of_charge += charge * ROUND_TRIP_EFFICIENCY
            value_gbp += charge * NOMINAL_PRICE_GBP_MWH
        elif state_of_charge > 0:
            discharge = min(BATTERY_POWER_MW * 0.5, state_of_charge)
            state_of_charge -= discharge

    return value_gbp


def top_n_periods(signal: pd.Series, n: int) -> pd.Series:
    """Mark the n periods with the highest signal as charge periods."""
    top = signal.nlargest(n).index
    return pd.Series(signal.index.isin(top), index=signal.index)


def battery_strategies(test: pd.DataFrame, predicted: pd.Series) -> dict:
    """Compare four dispatch strategies on an identical charging budget.

    Budget: top half of periods by each strategy's own signal.
    Perfect foresight uses actual curtailment — it is the ceiling.
    """
    actual = test[TARGET]
    budget = len(actual) // 2

    fixed_signal = pd.Series(
        test.index.hour.isin(range(1, 6)).astype(float), index=test.index
    )

    return {
        "no_forecast": simulate_battery(actual, top_n_periods(fixed_signal, budget)),
        "persistence": simulate_battery(
            actual, top_n_periods(test["curtailment_lag_2d"].fillna(0), budget)
        ),
        "model": simulate_battery(actual, top_n_periods(predicted, budget)),
        "perfect_foresight": simulate_battery(actual, top_n_periods(actual, budget)),
    }