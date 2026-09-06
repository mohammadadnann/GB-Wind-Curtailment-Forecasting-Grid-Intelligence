"""Dispatch value evaluation across all held out test folds.

Measures the £ value of correctly identifying curtailed periods.
Each strategy selects the same number of periods as the model.
Perfect foresight identifies all curtailed periods.
Value measured at £60/MWh nominal curtailment price.

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

        print(
            f"Fold {fold.number} ({fold.test_start.date()}): "
            f"model=£{strategies['model_gbp']:,.0f}  "
            f"perfect=£{strategies['perfect_foresight_gbp']:,.0f}  "
            f"capture={strategies['model_capture_pct']:.1f}%"
        )

    results = pd.DataFrame(rows)
    totals_model = results["model_gbp"].sum()
    totals_perfect = results["perfect_foresight_gbp"].sum()
    totals_persistence = results["persistence_gbp"].sum()
    totals_no_forecast = results["no_forecast_gbp"].sum()

    days_covered = sum(
        (evaluate.split(frame, f)[1].index.max() -
         evaluate.split(frame, f)[1].index.min()).days
        for f in folds
    )
    years_covered = days_covered / 365.25
    fleet_mw = 11_608

    print("\nTotal £ value captured across all test folds:")
    print(f"  no_forecast:       £{totals_no_forecast:,.0f}")
    print(f"  persistence:       £{totals_persistence:,.0f}")
    print(f"  model:             £{totals_model:,.0f}")
    print(f"  perfect_foresight: £{totals_perfect:,.0f}")

    pct = totals_model / totals_perfect * 100
    print(f"\nModel captures {pct:.1f}% of perfect foresight value")

    print(f"\nAnnualised per MW of Scottish fleet ({fleet_mw:,} MW):")
    for name, total in [
        ("no_forecast", totals_no_forecast),
        ("persistence", totals_persistence),
        ("model", totals_model),
        ("perfect_foresight", totals_perfect),
    ]:
        per_mw = total / fleet_mw / years_covered
        print(f"  {name:20s} £{per_mw:,.0f}/MW/year")

    results.to_csv(config.PROCESSED / "battery_results.csv", index=False)
    print(f"\nWrote battery_results.csv")


if __name__ == "__main__":
    main()