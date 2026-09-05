"""Curtailment dispatch evaluation across all held-out test folds.

Measures how well each strategy identifies constrained periods.
Precision: of periods the strategy selects, what fraction are actually constrained?
Recall: of constrained periods, what fraction does the strategy select?
F1: harmonic mean of precision and recall.

Run:  python scripts/13_battery_backtest.py
"""

import pandas as pd
import numpy as np

from src import config, evaluate, train


def evaluate_strategy(actual: pd.Series, signal: pd.Series, budget: int) -> dict:
    """Evaluate a dispatch strategy using precision, recall and F1."""
    selected = signal.nlargest(budget).index
    actually_constrained = actual[actual > 0].index

    true_positive = len(set(selected) & set(actually_constrained))
    precision = true_positive / budget if budget > 0 else 0.0
    recall = true_positive / len(actually_constrained) if len(actually_constrained) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    curtailment_captured = actual[actual.index.isin(selected)].sum()

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "curtailment_captured_mwh": round(curtailment_captured, 1),
    }


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    rows = []
    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)
        classifier, regressor = train.fit_hurdle(train_frame)
        predicted = train.predict_hurdle(classifier, regressor, test_frame)

        actual = test_frame[train.TARGET]
        budget = int((actual > 0).sum())

        fixed_signal = pd.Series(
            test_frame.index.hour.isin(range(1, 6)).astype(float),
            index=test_frame.index,
        )

        strategies = {
            "no_forecast": evaluate_strategy(actual, fixed_signal, budget),
            "persistence": evaluate_strategy(
                actual, test_frame["curtailment_lag_2d"].fillna(0), budget
            ),
            "model": evaluate_strategy(actual, predicted, budget),
            "perfect_foresight": evaluate_strategy(actual, actual, budget),
        }

        for name, metrics in strategies.items():
            rows.append({
                "fold": fold.number,
                "test_start": fold.test_start.date(),
                "strategy": name,
                **metrics,
            })

        model_f1 = strategies["model"]["f1"]
        perfect_f1 = strategies["perfect_foresight"]["f1"]
        print(
            f"Fold {fold.number} ({fold.test_start.date()}): "
            f"model F1={model_f1:.3f}  "
            f"perfect F1={perfect_f1:.3f}  "
            f"capture={model_f1/perfect_f1*100:.1f}%"
        )

    results = pd.DataFrame(rows)

    print("\nMean across folds by strategy:")
    summary = (
        results.groupby("strategy")[["precision", "recall", "f1", "curtailment_captured_mwh"]]
        .mean()
        .round(3)
        .loc[["no_forecast", "persistence", "model", "perfect_foresight"]]
    )
    print(summary.to_string())

    model_f1 = summary.loc["model", "f1"]
    perfect_f1 = summary.loc["perfect_foresight", "f1"]
    print(f"\nModel captures {model_f1/perfect_f1*100:.1f}% of perfect foresight F1")

    results.to_csv(config.PROCESSED / "dispatch_results.csv", index=False)
    print(f"\nWrote dispatch_results.csv")


if __name__ == "__main__":
    main()