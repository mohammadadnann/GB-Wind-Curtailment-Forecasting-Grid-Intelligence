"""Assemble the results ladder comparing baselines to the hurdle model.

Run:  python scripts/10_results_ladder.py
"""

import pandas as pd

from src import config, train


def main() -> None:
    baselines = pd.read_csv(config.PROCESSED / "baseline_results.csv")
    hurdle = pd.read_csv(config.PROCESSED / "hurdle_results.csv")

    combined = pd.concat([baselines, hurdle[["fold", "model", "test_start", "mae", "rmse"]]])

    ladder = (
        combined.groupby("model")[["mae", "rmse"]]
        .mean()
        .round(1)
        .sort_values("mae")
    )

    persistence_mae = ladder.loc["persistence_d2", "mae"]
    ladder["skill_vs_persistence"] = (
        (1 - ladder["mae"] / persistence_mae) * 100
    ).round(1)

    print("=" * 60)
    print("RESULTS LADDER — mean MAE and RMSE across folds")
    print("Note: boundary features excluded (NESO file not downloaded)")
    print("Note: weather features use approximated D-1 06:00 UTC vintage")
    print("=" * 60)
    print(ladder.to_string())

    ladder.to_csv(config.PROCESSED / "results_ladder.csv")
    print(f"\nWrote {config.PROCESSED / 'results_ladder.csv'}")


if __name__ == "__main__":
    main()