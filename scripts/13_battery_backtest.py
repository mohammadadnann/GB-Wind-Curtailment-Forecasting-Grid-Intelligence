"""Battery dispatch simulation across all held-out test folds.

A 50 MW / 200 MWh battery is dispatched under four strategies, each given
an identical charging budget of the top half of test periods by its own signal.
Perfect foresight uses actual curtailment as its signal.

Note: individual folds and the no_forecast strategy may show values above
perfect foresight due to battery capacity dynamics. This is a known simulation
mechanic disclosed in the results rather than hidden.

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
            f"model=£{strategies['model']:,.0f}  "
            f"perfect=£{strategies['perfect_foresight']:,.0f}  "
            f"capture={pct:.1f}%"
        )

    results = pd.DataFrame(rows)
    value_cols = ["no_forecast", "persistence", "model", "perfect_foresight"]
    totals = results[value_cols].sum()

    print("\nTotal £ value captured across all test folds:")
    print(totals.round(0).to_string())

    days_covered = (
        results["test_start"].max() - results["test_start"].min()
    ).days + 90
    years_covered = days_covered / 365.25
    per_mw_year = (totals / train.BATTERY_POWER_MW / years_covered).round(0)

    print(f"\nApproximate coverage: {years_covered:.2f} years")
    print("\nGBP per MW per year:")
    print(per_mw_year.to_string())

    pct_of_perfect = (totals["model"] / totals["perfect_foresight"] * 100).round(1)
    print(f"\nModel captures {pct_of_perfect}% of perfect foresight value")
    if pct_of_perfect > 100:
        print("Note: model exceeds perfect foresight aggregate due to battery")
        print("capacity dynamics. See script docstring for explanation.")

    results.to_csv(config.PROCESSED / "battery_results.csv", index=False)
    print(f"\nWrote battery_results.csv")


if __name__ == "__main__":
    main()