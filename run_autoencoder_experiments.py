from __future__ import annotations

import pandas as pd

from autoencoder import AEConfig, run_single_autoencoder
from ai_common import per_intent_recall

RULE_VAL_PATH = "val_rule_results.csv"
RULE_TEST_PATH = "test_rule_results.csv"

TARGET_RECALL = 0.85
SEEDS = [42, 52, 62]

SUMMARY_OUTPUT_PATH = "ae_experiment_summary.csv"
PER_INTENT_OUTPUT_PATH = "ae_per_intent_summary.csv"
COMPARISON_OUTPUT_PATH = "ae_vs_rules_comparison.csv"


def average_metric_rows(rows: list[dict], prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "metric": df.columns,
        f"{prefix}_mean": df.mean().values,
        f"{prefix}_std": df.std(ddof=0).values,
    })
    return summary


def compare_with_rules() -> pd.DataFrame:
    rule_val = pd.read_csv(RULE_VAL_PATH, parse_dates=["timestamp"])
    rule_test = pd.read_csv(RULE_TEST_PATH, parse_dates=["timestamp"])

    rule_val_intent = per_intent_recall(rule_val, alert_col="rule_alert")
    rule_val_intent.rename(columns={"recall": "rules_recall"}, inplace=True)
    rule_val_intent["dataset"] = "val"

    rule_test_intent = per_intent_recall(rule_test, alert_col="rule_alert")
    rule_test_intent.rename(columns={"recall": "rules_recall"}, inplace=True)
    rule_test_intent["dataset"] = "test"

    return pd.concat([rule_val_intent, rule_test_intent], ignore_index=True)


def main() -> None:
    print("Running Autoencoder experiments...")
    print(f"Target recall on validation: {TARGET_RECALL}")
    print(f"Seeds: {SEEDS}")

    config = AEConfig()

    all_val_metrics = []
    all_test_metrics = []
    all_intents = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = run_single_autoencoder(
            config=config,
            seed=seed,
            target_recall=TARGET_RECALL,
        )

        print(f"Threshold selected: {result['threshold']:.6f}")
        print(f"Epochs ran: {result['train_info']['epochs_ran']}")
        print(f"Best inner-val loss: {result['train_info']['best_val_loss']:.6f}")

        print("Validation metrics:")
        for k, v in result["val_metrics"].items():
            print(f"  {k}: {v}")

        print("Test metrics:")
        for k, v in result["test_metrics"].items():
            print(f"  {k}: {v}")

        all_val_metrics.append(result["val_metrics"])
        all_test_metrics.append(result["test_metrics"])
        all_intents.append(result["val_intent"])
        all_intents.append(result["test_intent"])

    val_summary = average_metric_rows(all_val_metrics, prefix="val")
    test_summary = average_metric_rows(all_test_metrics, prefix="test")
    summary = val_summary.merge(test_summary, on="metric", how="outer")
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    intent_df = pd.concat(all_intents, ignore_index=True)
    intent_summary = (
        intent_df
        .groupby(["dataset", "intent"], as_index=False)
        .agg(
            count_mean=("count", "mean"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .sort_values(["dataset", "intent"])
    )
    intent_summary.to_csv(PER_INTENT_OUTPUT_PATH, index=False)

    rules_df = compare_with_rules()
    comparison_df = intent_summary.merge(
        rules_df[["dataset", "intent", "rules_recall"]],
        on=["dataset", "intent"],
        how="left"
    )
    comparison_df["ae_better"] = comparison_df["recall_mean"] > comparison_df["rules_recall"]
    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("\n=== Autoencoder Summary ===")
    print(summary)

    print("\n=== Autoencoder Per-Intent Summary ===")
    print(intent_summary)

    print("\n=== Autoencoder vs Rules Comparison ===")
    print(comparison_df)

    print("\nSaved:")
    print(f"- {SUMMARY_OUTPUT_PATH}")
    print(f"- {PER_INTENT_OUTPUT_PATH}")
    print(f"- {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()