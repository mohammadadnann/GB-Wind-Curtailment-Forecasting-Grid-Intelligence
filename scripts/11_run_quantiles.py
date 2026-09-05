"""Run quantile models with proper train / calibration / test split.

The calibration slice is the most recent 15 percent of each training window.
Quantile models are fit on the remainder. The conformal widening factor is
derived from the calibration slice only — the test set is never touched
during calibration.

Run:  python scripts/11_run_quantiles.py
"""

import pandas as pd

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    rows = []
    for fold in folds:
        train_frame, test_frame = evaluate.split(frame, fold)

        # Split training into fit and calibration — calibration is never
        # seen by the quantile models during fitting
        split_point = int(len(train_frame) * 0.85)
        fit_frame = train_frame.iloc[:split_point]
        calib_frame = train_frame.iloc[split_point:]

        classifier, _ = train.fit_hurdle(fit_frame)
        quantile_models = train.fit_quantile_models(fit_frame)
        widening = train.fit_conformal_widening(
            classifier, quantile_models, calib_frame
        )

        predicted = train.predict_quantiles(classifier, quantile_models, test_frame)
        predicted = train.apply_conformal_widening(predicted, widening)

        actual = test_frame[train.TARGET]
        cov = train.coverage(actual, predicted["p10"], predicted["p90"])

        rows.append({
            "fold": fold.number,
            "test_start": fold.test_start.date(),
            "widening_factor": widening,
            "pinball_p10": round(train.pinball_loss(actual, predicted["p10"], 0.1), 2),
            "pinball_p50": round(train.pinball_loss(actual, predicted["p50"], 0.5), 2),
            "pinball_p90": round(train.pinball_loss(actual, predicted["p90"], 0.9), 2),
            "coverage_pct": round(cov * 100, 1),
        })

        print(
            f"Fold {fold.number} ({fold.test_start.date()}): "
            f"coverage={cov:.1%} widening={widening:.3f}"
        )

    results = pd.DataFrame(rows)
    print("\nMean across folds:")
    print(
        results[["pinball_p10", "pinball_p50", "pinball_p90", "coverage_pct"]]
        .mean().round(2).to_string()
    )
    print(f"Mean widening factor: {results['widening_factor'].mean():.3f}")

    results.to_csv(config.PROCESSED / "quantile_results.csv", index=False)
    print(f"\nWrote quantile_results.csv")


if __name__ == "__main__":
    main()