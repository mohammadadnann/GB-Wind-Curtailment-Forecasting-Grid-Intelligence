"""Train final models on the full dataset and save artefacts.

The final models are trained on all available data (no test holdout) since
they are intended for production inference, not evaluation. Evaluation
metrics come from the rolling-origin backtest in scripts 08-13.

Artefacts saved:
  models/classifier.joblib      LightGBM classifier
  models/regressor.joblib       LightGBM volume regressor
  models/quantile_p10.joblib    Quantile regressor P10
  models/quantile_p50.joblib    Quantile regressor P50
  models/quantile_p90.joblib    Quantile regressor P90
  models/artefact_card.json     Feature list, training cut-off, version

Run:  python scripts/14_save_artefacts.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src import config, train

MODELS_DIR = config.ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    clean = frame.dropna(subset=train.FEATURE_COLUMNS + [train.TARGET])

    print(f"Training on {len(clean):,} periods "
          f"({clean.index.min().date()} to {clean.index.max().date()})")

    classifier, regressor = train.fit_hurdle(clean)
    quantile_models = train.fit_quantile_models(clean)

    joblib.dump(classifier, MODELS_DIR / "classifier.joblib")
    joblib.dump(regressor, MODELS_DIR / "regressor.joblib")
    for q, model in quantile_models.items():
        joblib.dump(model, MODELS_DIR / f"quantile_p{int(q * 100)}.joblib")

    card = {
        "version": "1.0.0",
        "training_cutoff": str(clean.index.max().date()),
        "training_periods": len(clean),
        "feature_columns": train.FEATURE_COLUMNS,
        "target": train.TARGET,
        "model_architecture": "two-stage hurdle: LightGBM classifier + conditional regressor",
        "quantiles": [0.1, 0.5, 0.9],
        "evaluation": {
            "mean_mae": 359.7,
            "mean_skill_vs_persistence": 0.428,
            "mean_f1_dispatch": 0.832,
            "folds": 6,
            "method": "expanding-window rolling-origin backtesting",
        },
        "notes": [
            "Boundary features excluded — NESO constraint limits file not downloaded.",
            "Weather features use approximated D-1 06:00 UTC NWP vintage.",
            "Cost model not included — requires BOD bid price data.",
            "Evaluation metrics from rolling-origin backtest in data/processed/.",
        ],
    }

    with open(MODELS_DIR / "artefact_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print("\nSaved artefacts:")
    for path in sorted(MODELS_DIR.iterdir()):
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:35s} {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()