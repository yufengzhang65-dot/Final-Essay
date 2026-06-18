from __future__ import annotations

import pandas as pd

from ocsvm import run_single_ocsvm
from ai_common import per_intent_recall

RULE_VAL_PATH = "val_rule_results.csv"
RULE_TEST_PATH = "test_rule_results.csv"

TARGET_RECALL = 0.85

OCSVM_PARAM_GRID = [
    {"nu": 0.03, "gamma": "scale"},
    {"nu": 0.05, "gamma": "scale"},
    {"nu": 0.08, "gamma": "scale"},
    {"nu": 0.03, "gamma": 0.05},
    {"nu": 0.05, "gamma": 0.05},
    {"nu": 0.08, "gamma": 0.05},
]

SUMMARY_OUTPUT_PATH = "ocsvm_experiment_summary.csv"
PER_INTENT_OUTPUT_PATH = "ocsvm_per_intent_summary.csv"
COMPARISON_OUTPUT_PATH = "ocsvm_vs_rules_comparison.csv"


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
    print("Running OCSVM parameter experiments...")
    print(f"Target recall on validation: {TARGET_RECALL}")

    summary_rows = []
    intent_rows = []

    for params in OCSVM_PARAM_GRID:
        print(f"\n--- nu={params['nu']}, gamma={params['gamma']} ---")
        result = run_single_ocsvm(
            nu=params["nu"],
            gamma=params["gamma"],
            target_recall=TARGET_RECALL,
        )

        val_metrics = result["val_metrics"]
        test_metrics = result["test_metrics"]

        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

        summary_rows.append({
            "nu": params["nu"],
            "gamma": params["gamma"],
            "threshold": result["threshold"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_alerts_per_1000": val_metrics["alerts_per_1000"],
            "val_time_to_detect": val_metrics["mean_time_to_detect_sec"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_alerts_per_1000": test_metrics["alerts_per_1000"],
            "test_time_to_detect": test_metrics["mean_time_to_detect_sec"],
        })

        val_intent = result["val_intent"].copy()
        val_intent["nu"] = params["nu"]
        val_intent["gamma"] = params["gamma"]

        test_intent = result["test_intent"].copy()
        test_intent["nu"] = params["nu"]
        test_intent["gamma"] = params["gamma"]

        intent_rows.append(val_intent)
        intent_rows.append(test_intent)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    intent_df = pd.concat(intent_rows, ignore_index=True)
    intent_df.to_csv(PER_INTENT_OUTPUT_PATH, index=False)

    rules_df = compare_with_rules()

    comparison_df = intent_df.merge(
        rules_df[["dataset", "intent", "rules_recall"]],
        on=["dataset", "intent"],
        how="left"
    )
    comparison_df["ocsvm_better"] = comparison_df["recall"] > comparison_df["rules_recall"]
    comparison_df.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("\n=== OCSVM Summary ===")
    print(summary_df)

    print("\n=== OCSVM vs Rules Comparison ===")
    print(comparison_df)

    print("\nSaved:")
    print(f"- {SUMMARY_OUTPUT_PATH}")
    print(f"- {PER_INTENT_OUTPUT_PATH}")
    print(f"- {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()