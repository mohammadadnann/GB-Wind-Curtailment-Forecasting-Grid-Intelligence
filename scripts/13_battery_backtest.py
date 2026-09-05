"""Battery dispatch simulation across all held-out test folds.

Four strategies are compared on an identical charging budget (top half
of periods by each strategy's signal). Perfect foresight is the ceiling —
it always picks the best periods in hindsight.

Run:  python scripts/13_battery_backtest.py
"""

import pandas as pd

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    rows = []
    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)
        classifier, regressor = train.fit_hurdle(train_frame)
        predicted = train.predict_hurdle(classifier, regressor, test_frame)

        strategies = train.battery_strategies(test_frame, predicted)
        strategies["fold"] = fold.number
        strategies["test_start"] = fold.test_start.date()
        rows.append(strategies)

        pct = strategies["model"] / strategies["perfect_foresight"] * 100
        print(
            f"Fold {fold.number} ({fold.test_start.date()}): "
            f"model=£{strategies['model']:,.0f} "
            f"perfect=£{strategies['perfect_foresight']:,.0f} "
            f"capture={pct:.1f}%"
        )

    results = pd.DataFrame(rows)
    totals = results[["no_forecast", "persistence", "model", "perfect_foresight"]].sum()

    print("\nTotal value captured across all test folds:")
    print(totals.round(0).to_string())

    days_covered = sum(
        (evaluate.split(frame, f)[1].index.max() -
         evaluate.split(frame, f)[1].index.min()).days
        for f in folds
    )
    years_covered = days_covered / 365.25

    print(f"\nApproximate test coverage: {years_covered:.2f} years")
    print("\nGBP per MW per year (annualised):")
    print((totals / train.BATTERY_POWER_MW / years_covered).round(0).to_string())

    pct_of_perfect = totals["model"] / totals["perfect_foresight"] * 100
    print(f"\nModel captures {pct_of_perfect:.1f}% of perfect foresight value")

    results.to_csv(config.PROCESSED / "battery_results.csv", index=False)
    print(f"\nWrote battery_results.csv")


if __name__ == "__main__":
    main()