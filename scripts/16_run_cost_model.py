"""Running the cost model across all expanding-window folds.

Run:  python scripts/16_run_cost_model.py
"""

import pandas as pd

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    rows = []
    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)

        classifier, _ = train.fit_hurdle(train_frame)

        try:
            cost_model = train.fit_cost_model(train_frame)
        except ValueError as e:
            print(f"Fold {fold.number}: {e}")
            continue

        predicted_cost = train.predict_cost(classifier, cost_model, test_frame)
        actual_cost = test_frame["curtailment_cost_gbp"]

        fold_mae = train.mae(actual_cost, predicted_cost)
        fold_rmse = train.rmse(actual_cost, predicted_cost)

        actual_total = actual_cost.sum()
        predicted_total = predicted_cost.sum()

        rows.append({
            "fold": fold.number,
            "test_start": fold.test_start.date(),
            "mae_gbp": round(fold_mae, 0),
            "rmse_gbp": round(fold_rmse, 0),
            "actual_total_gbpm": round(actual_total / 1e6, 1),
            "predicted_total_gbpm": round(predicted_total / 1e6, 1),
        })

        print(
            f"Fold {fold.number} ({fold.test_start.date()}): "
            f"MAE=£{fold_mae:,.0f}  "
            f"actual=£{actual_total/1e6:.1f}m  "
            f"predicted=£{predicted_total/1e6:.1f}m"
        )

    results = pd.DataFrame(rows)
    print(f"\nMean MAE: £{results['mae_gbp'].mean():,.0f}/period")
    print(f"Total actual: £{results['actual_total_gbpm'].sum():.0f}m")
    print(f"Total predicted: £{results['predicted_total_gbpm'].sum():.0f}m")

    results.to_csv(config.PROCESSED / "cost_model_results.csv", index=False)
    print(f"\nWrote cost_model_results.csv")


if __name__ == "__main__":
    main()