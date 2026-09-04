"""Run the four baselines across all expanding-window folds.

Run:  python scripts/08_run_baselines.py
"""

import pandas as pd

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    print(evaluate.describe_folds(folds).to_string(index=False))
    print()

    rows = []
    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)

        predictions = {
            "zero": train.baseline_zero(test_frame),
            "persistence_d2": train.baseline_persistence(train_frame, test_frame),
            "weekly_d7": train.baseline_weekly(train_frame, test_frame),
            "physical": train.baseline_physical(test_frame),
        }

        actual = test_frame[train.TARGET]
        for name, predicted in predictions.items():
            rows.append({
                "fold": fold.number,
                "model": name,
                "test_start": fold.test_start.date(),
                "mae": round(train.mae(actual, predicted), 1),
                "rmse": round(train.rmse(actual, predicted), 1),
            })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    print("\nMean across folds:")
    print(results.groupby("model")[["mae", "rmse"]].mean().round(1).sort_values("mae").to_string())

    out = config.PROCESSED / "baseline_results.csv"
    results.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()