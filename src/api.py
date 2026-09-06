"""FastAPI service for the GB curtailment forecast.

It serves forecasts for dates within the backtest window. It does not construct tomorrow's features
from live data sources. Every response includes a clear label stating this.

Models are loaded once at startup from saved joblib files. The service
never retrains. If files are missing, startup fails with a clear message.

"""

from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config, train

MODELS_DIR = config.ROOT / "models"

app = FastAPI(
    title="GB Constraint Cost Intelligence",
    description=(
        "Historical demonstration of day-ahead Scottish wind curtailment forecasting. "
        "Serves backtest dates only — not a live day-ahead system."
    ),
    version="1.0.0",
)

MODELS: dict = {}


def load_artefacts() -> None:
    """Load all saved model artefacts from disk."""
    required = [
        "classifier.joblib",
        "regressor.joblib",
        "quantile_p10.joblib",
        "quantile_p50.joblib",
        "quantile_p90.joblib",
        "artefact_card.json",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing model artefacts: {missing}. "
            "Run scripts/14_save_artefacts.py first."
        )

    MODELS["classifier"] = joblib.load(MODELS_DIR / "classifier.joblib")
    MODELS["regressor"] = joblib.load(MODELS_DIR / "regressor.joblib")
    MODELS["quantile_models"] = {
        0.1: joblib.load(MODELS_DIR / "quantile_p10.joblib"),
        0.5: joblib.load(MODELS_DIR / "quantile_p50.joblib"),
        0.9: joblib.load(MODELS_DIR / "quantile_p90.joblib"),
    }

    with open(MODELS_DIR / "artefact_card.json") as f:
        MODELS["card"] = json.load(f)

    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    MODELS["frame"] = frame
    MODELS["available_dates"] = sorted(set(frame.index.date))


@app.on_event("startup")
def startup() -> None:
    load_artefacts()


@app.get("/health")
def health() -> dict:
    """Service and model status."""
    if not MODELS:
        return {"status": "degraded", "models_loaded": False}
    return {
        "status": "ok",
        "models_loaded": True,
        "training_cutoff": MODELS["card"]["training_cutoff"],
        "available_dates": {
            "first": str(MODELS["available_dates"][0]),
            "last": str(MODELS["available_dates"][-1]),
        },
        "note": (
            "Historical demonstration only. "
            "Forecasts are served from the backtest feature table, "
            "not from live data sources."
        ),
    }


class PeriodForecast(BaseModel):
    datetime_utc: str
    settlement_period: int
    p_constrained: float
    curtailed_mwh_p10: float
    curtailed_mwh_p50: float
    curtailed_mwh_p90: float


class ForecastResponse(BaseModel):
    date: str
    system_label: str
    total_curtailed_mwh_p50: float
    periods: list[PeriodForecast]



@app.get("/forecast", response_model=ForecastResponse)
def forecast(target_date: date) -> ForecastResponse:
    """Return point and quantile forecast for every settlement period of one date.

    Only dates within the backtest feature table are available.
    """
    if target_date not in MODELS["available_dates"]:
        raise HTTPException(
            404,
            f"No feature data for {target_date}. "
            f"Available range: {MODELS['available_dates'][0]} "
            f"to {MODELS['available_dates'][-1]}.",
        )

    frame = MODELS["frame"]
    day = frame[frame.index.date == target_date]

    classifier = MODELS["classifier"]
    quantile_models = MODELS["quantile_models"]

    prob = classifier.predict_proba(day[train.FEATURE_COLUMNS])[:, 1]
    quantiles = train.predict_quantiles(classifier, quantile_models, day)

    periods = []
    for i, (ts, _) in enumerate(day.iterrows()):
        sp = int(ts.hour * 2 + ts.minute // 30 + 1)
        periods.append(PeriodForecast(
            datetime_utc=ts.isoformat(),
            settlement_period=sp,
            p_constrained=round(float(prob[i]), 3),
            curtailed_mwh_p10=round(float(quantiles["p10"].iloc[i]), 1),
            curtailed_mwh_p50=round(float(quantiles["p50"].iloc[i]), 1),
            curtailed_mwh_p90=round(float(quantiles["p90"].iloc[i]), 1),
        ))

    return ForecastResponse(
        date=str(target_date),
        system_label="historical_demonstration",
        total_curtailed_mwh_p50=round(float(quantiles["p50"].sum()), 1),
        periods=periods,
    )


@app.get("/explain")
def explain(target_date: date, period: int) -> dict:
    """Return top SHAP contributions for one settlement period."""
    import shap

    if target_date not in MODELS["available_dates"]:
        raise HTTPException(404, f"No feature data for {target_date}.")

    frame = MODELS["frame"]
    day = frame[frame.index.date == target_date]
    matching = day[day.index.hour * 2 + day.index.minute // 30 + 1 == period]

    if matching.empty:
        raise HTTPException(404, f"Period {period} not found for {target_date}.")

    row = matching[train.FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(MODELS["regressor"])
    shap_values = explainer.shap_values(row)

    contributions = (
        pd.Series(shap_values[0], index=train.FEATURE_COLUMNS)
        .sort_values(key=abs, ascending=False)
        .head(10)
    )

    return {
        "date": str(target_date),
        "period": period,
        "system_label": "historical_demonstration",
        "base_value": round(float(explainer.expected_value), 3),
        "top_contributions": [
            {"feature": name, "shap_value": round(float(val), 4)}
            for name, val in contributions.items()
        ],
    }