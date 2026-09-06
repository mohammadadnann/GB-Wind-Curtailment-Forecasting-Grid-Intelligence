"""Forecasting models for Scottish wind curtailment

I am using a two stage approach:
  Stage 1 — a classifier that predicts whether a period will have any curtailment at all
  Stage 2 — a regressor that predicts how much curtailment (in MWh), trained only on
             periods that were actually curtailed

Combining both stages: predicted MWh = P(curtailed) × predicted volume if curtailed.

The volume target is log transformed before training to handle the long tail of
very large curtailment events, then converted back when making predictions.

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
    "b6_limit_mw",
    "b6_limit_vs_30d_median",
    "b6_outage_flag",
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
    """Estimate curtailment from wind speed using a cubic power curve."""
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
    """Fit the two-stage hurdle model."""
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


def predict_hurdle(classifier, regressor, test: pd.DataFrame) -> pd.Series:
    """Combine classifier and regressor into a point forecast."""
    X = test[FEATURE_COLUMNS]
    prob = classifier.predict_proba(X)[:, 1]
    log_volume = regressor.predict(X)
    volume = np.expm1(log_volume)
    return pd.Series(prob * volume, index=test.index)


# --- quantile models ---

QUANTILES = [0.1, 0.5, 0.9]


def fit_quantile_models(train: pd.DataFrame) -> dict:
    """Fit one quantile regressor per quantile on constrained periods only."""
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
    """Combine classifier with quantile regressors using zero-inflated mixture."""
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
    """Find the smallest widening factor achieving target coverage on calibration slice."""
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
    """Apply calibrated widening factor to quantile predictions."""
    result = predicted.copy()
    result["p10"] = (predicted["p50"] - (predicted["p50"] - predicted["p10"]) * factor).clip(lower=0)
    result["p90"] = predicted["p50"] + (predicted["p90"] - predicted["p50"]) * factor
    return result


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    """Pinball loss — correct scoring rule for a quantile forecast."""
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


def fit_cost_model(train: pd.DataFrame):
    """Fit a LightGBM regressor for constraint cost in pounds."""
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


# --- dispatch value simulation ---

NOMINAL_PRICE_GBP_MWH = 60.0


def dispatch_value(
    curtailed_mwh: pd.Series,
    signal: pd.Series,
    n_select: int,
) -> dict:
    """Measure the £ value of acting on the top-n periods by signal.

    Selects the n periods with the highest signal. For each selected period
    where curtailment actually occurs, the value captured is the actual
    curtailed MWh at the nominal price. Perfect foresight uses actual
    curtailment as its signal and is always the ceiling.
    """
    selected = signal.nlargest(n_select).index
    actual_selected = curtailed_mwh.reindex(selected).fillna(0.0)
    true_positive_mwh = actual_selected[actual_selected > 0].sum()
    value_gbp = true_positive_mwh * NOMINAL_PRICE_GBP_MWH

    pf_selected = curtailed_mwh.nlargest(n_select).index
    pf_mwh = curtailed_mwh.reindex(pf_selected).fillna(0.0).sum()
    pf_gbp = pf_mwh * NOMINAL_PRICE_GBP_MWH

    return {
        "value_gbp": value_gbp,
        "pf_gbp": pf_gbp,
        "capture_pct": value_gbp / pf_gbp * 100 if pf_gbp > 0 else 0.0,
        "true_positive_mwh": true_positive_mwh,
        "pf_mwh": pf_mwh,
    }


def battery_strategies(test: pd.DataFrame, predicted: pd.Series) -> dict:
    """Compare four dispatch strategies on an identical selection budget.

    Budget equals the number of constrained periods in the test set.
    Each strategy selects that many periods by its own signal.
    Perfect foresight selects the top-n by actual curtailment — ceiling.
    Value measured at £60/MWh nominal curtailment price.
    """
    actual = test[TARGET]
    budget = int((actual > 0).sum())

    fixed_signal = pd.Series(
        test.index.hour.isin(range(1, 6)).astype(float), index=test.index
    )
    persistence_signal = test["curtailment_lag_2d"].fillna(0)

    no_forecast = dispatch_value(actual, fixed_signal, budget)
    persistence = dispatch_value(actual, persistence_signal, budget)
    model = dispatch_value(actual, predicted, budget)
    perfect = dispatch_value(actual, actual, budget)

    return {
        "no_forecast_gbp": no_forecast["value_gbp"],
        "persistence_gbp": persistence["value_gbp"],
        "model_gbp": model["value_gbp"],
        "perfect_foresight_gbp": perfect["value_gbp"],
        "model_capture_pct": model["capture_pct"],
        "model_tp_mwh": model["true_positive_mwh"],
        "pf_mwh": perfect["pf_mwh"],
    }