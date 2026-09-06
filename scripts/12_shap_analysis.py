"""SHAP analysis on held out test examples from the final fold.

Global importance shows which features drive the model on average.
Local explanation shows why the model predicted a specific value.
Both use examples from the held-out test set — never training data.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config, evaluate, train


def main() -> None:
    frame = pd.read_parquet(config.PROCESSED / "features.parquet")
    folds = evaluate.build_folds(frame.index)

    # Using the final fold for SHAP — it has the most training data
    fold = folds[-1]
    train_frame, test_frame = evaluate.split(frame, fold)

    classifier, regressor = train.fit_hurdle(train_frame)
    predicted = train.predict_hurdle(classifier, regressor, test_frame)

    print(f"SHAP analysis on fold {fold.number}: "
          f"test {fold.test_start.date()} to {fold.test_end.date()}")
    print(f"Test periods: {len(test_frame):,}")

    # Global importance — classifier
    X_test = test_frame[train.FEATURE_COLUMNS]
    explainer_class = shap.TreeExplainer(classifier)
    shap_class = explainer_class.shap_values(X_test)
    if isinstance(shap_class, list):
        shap_class = shap_class[1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_class, X_test, plot_type="bar", show=False)
    plt.title("Feature importance — P(constrained) classifier")
    plt.tight_layout()
    plt.savefig(config.ROOT / "docs" / "shap_classifier_importance.png", dpi=150)
    plt.close()

    # Global importance — regressor (constrained periods only)
    constrained_test = test_frame[test_frame["is_constrained"] == 1]
    X_constrained = constrained_test[train.FEATURE_COLUMNS]
    explainer_reg = shap.TreeExplainer(regressor)
    shap_reg = explainer_reg.shap_values(X_constrained)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_reg, X_constrained, plot_type="bar", show=False)
    plt.title("Feature importance — curtailed MWh regressor (constrained periods)")
    plt.tight_layout()
    plt.savefig(config.ROOT / "docs" / "shap_regressor_importance.png", dpi=150)
    plt.close()

    # Print top features
    class_imp = pd.Series(
        np.abs(shap_class).mean(axis=0), index=train.FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\nTop 10 — classifier:")
    print(class_imp.head(10).round(4).to_string())

    reg_imp = pd.Series(
        np.abs(shap_reg).mean(axis=0), index=train.FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\nTop 10 — regressor:")
    print(reg_imp.head(10).round(4).to_string())

    # Local explanation — highest error period from held-out test
    errors = (test_frame[train.TARGET] - predicted).abs()
    worst_time = errors.idxmax()
    actual_value = test_frame.loc[worst_time, train.TARGET]
    pred_value = predicted[worst_time]

    print(f"\nLocal explanation for worst error:")
    print(f"  Time: {worst_time}")
    print(f"  Actual: {actual_value:.0f} MWh, Predicted: {pred_value:.0f} MWh")

    if worst_time in X_constrained.index:
        row = X_constrained.loc[[worst_time]]
        row_shap = explainer_reg.shap_values(row)
        contributions = pd.Series(
            row_shap[0], index=train.FEATURE_COLUMNS
        ).sort_values(key=abs, ascending=False)

        print("\n  Top 10 SHAP contributions (log scale):")
        for feat, val in contributions.head(10).items():
            direction = "UP  " if val > 0 else "DOWN"
            print(f"    {feat:35s} {direction} {abs(val):.4f}  "
                  f"(value={row[feat].iloc[0]:.3f})")

        explanation = shap.Explanation(
            values=row_shap[0],
            base_values=explainer_reg.expected_value,
            data=row.iloc[0].values,
            feature_names=train.FEATURE_COLUMNS,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Local explanation — {worst_time.date()} period {worst_time.time()}")
        plt.tight_layout()
        plt.savefig(config.ROOT / "docs" / "shap_local_waterfall.png", dpi=150)
        plt.close()
        print("\nSaved shap_local_waterfall.png")

    print("\nSaved shap_classifier_importance.png")
    print("Saved shap_regressor_importance.png")


if __name__ == "__main__":
    main()