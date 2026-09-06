"""Run the hurdle model across all expanding-window folds.

Run:  python scripts/09_run_hurdle.py
"""

import pandas as pd

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    rows = []
    all_predictions = []

    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)

        classifier, regressor = train.fit_hurdle(train_frame)
        predicted = train.predict_hurdle(classifier, regressor, test_frame)

        actual = test_frame[train.TARGET]
        fold_mae = train.mae(actual, predicted)
        fold_rmse = train.rmse(actual, predicted)

        baselines = pd.read_csv(config.PROCESSED / "baseline_results.csv")
        persistence_mae = (
            baselines[
                (baselines["fold"] == fold.number)
                & (baselines["model"] == "persistence_d2")
            ]["mae"].values[0]
        )

        rows.append({
            "fold": fold.number,
            "model": "hurdle",
            "test_start": fold.test_start.date(),
            "mae": round(fold_mae, 1),
            "rmse": round(fold_rmse, 1),
            "skill_vs_persistence": round(
                train.skill_score(fold_mae, persistence_mae), 3
            ),
        })

        pred_frame = test_frame[[train.TARGET]].copy()
        pred_frame["predicted"] = predicted
        pred_frame["fold"] = fold.number
        all_predictions.append(pred_frame)

        print(f"Fold {fold.number} ({fold.test_start.date()}): "
              f"MAE={fold_mae:.1f} RMSE={fold_rmse:.1f} "
              f"skill={train.skill_score(fold_mae, persistence_mae):.3f}")

    results = pd.DataFrame(rows)
    print(f"\nMean MAE: {results['mae'].mean():.1f}")
    print(f"Mean skill vs persistence: {results['skill_vs_persistence'].mean():.3f}")

    results.to_csv(config.PROCESSED / "hurdle_results.csv", index=False)

    predictions = pd.concat(all_predictions)
    predictions.to_parquet(config.PROCESSED / "hurdle_predictions.parquet")

    print("\nWrote hurdle_results.csv and hurdle_predictions.parquet")


if __name__ == "__main__":
    main()