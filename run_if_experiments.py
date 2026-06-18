from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ai_common import (
    load_feature_sets,
    prepare_model_inputs,
    select_threshold_by_target_recall,
    summarize_ai_metrics,
    per_intent_recall,
)

TRAIN_FEATURES_PATH = "train_normal_features.csv"
VAL_FEATURES_PATH = "val_features.csv"
TEST_FEATURES_PATH = "test_features.csv"

RULE_VAL_PATH = "val_rule_results.csv"
RULE_TEST_PATH = "test_rule_results.csv"

TARGET_RECALL = 0.85
SEEDS = [42, 52, 62]

SUMMARY_OUTPUT_PATH = "if_experiment_summary.csv"
PER_INTENT_OUTPUT_PATH = "if_per_intent_summary.csv"
COMPARISON_OUTPUT_PATH = "if_vs_rules_comparison.csv"


def fit_isolation_forest(X_train: np.ndarray, random_seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # Higher score = more anomalous
    return -model.decision_function(X)


def attach_outputs(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["if_score"] = scores
    out["if_alert"] = (out["if_score"] >= threshold).astype(int)
    return out


def run_single_seed(seed: int) -> dict:
    train_df, val_df, test_df = load_feature_sets(
        TRAIN_FEATURES_PATH,
        VAL_FEATURES_PATH,
        TEST_FEATURES_PATH,
    )

    X_train = prepare_model_inputs(train_df)
    X_val = prepare_model_inputs(val_df)
    X_test = prepare_model_inputs(test_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = fit_isolation_forest(X_train_scaled, random_seed=seed)

    val_scores = anomaly_scores(model, X_val_scaled)
    test_scores = anomaly_scores(model, X_test_scaled)

    threshold = select_threshold_by_target_recall(
        y_true=val_df["label"].astype(int).values,
        scores=val_scores,
        target_recall=TARGET_RECALL,
    )

    val_results = attach_outputs(val_df, val_scores, threshold)
    test_results = attach_outputs(test_df, test_scores, threshold)

    val_metrics = summarize_ai_metrics(val_results, score_col="if_score", alert_col="if_alert")
    test_metrics = summarize_ai_metrics(test_results, score_col="if_score", alert_col="if_alert")

    val_intent = per_intent_recall(val_results, alert_col="if_alert")
    val_intent["dataset"] = "val"
    val_intent["seed"] = seed

    test_intent = per_intent_recall(test_results, alert_col="if_alert")
    test_intent["dataset"] = "test"
    test_intent["seed"] = seed

    return {
        "seed": seed,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_intent": val_intent,
        "test_intent": test_intent,
    }


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
    print("Running Isolation Forest experiments...")
    print(f"Seeds: {SEEDS}")
    print(f"Target recall on validation: {TARGET_RECALL}")

    all_val_metrics = []
    all_test_metrics = []
    all_intent_tables = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = run_single_seed(seed)

        print(f"Threshold selected: {result['threshold']:.6f}")

        print("Validation metrics:")
        for k, v in result["val_metrics"].items():
            print(f"  {k}: {v}")

        print("Test metrics:")
        for k, v in result["test_metrics"].items():
            print(f"  {k}: {v}")

        all_val_metrics.append(result["val_metrics"])
        all_test_metrics.append(result["test_metrics"])
        all_intent_tables.append(result["val_intent"])
        all_intent_tables.append(result["test_intent"])

    # Summaries
    val_summary = average_metric_rows(all_val_metrics, prefix="val")
    test_summary = average_metric_rows(all_test_metrics, prefix="test")
    summary = val_summary.merge(test_summary, on="metric", how="outer")
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    # Per-intent average recall
    all_intents = pd.concat(all_intent_tables, ignore_index=True)
    intent_summary = (
        all_intents
        .groupby(["dataset", "intent"], as_index=False)
        .agg(
            count_mean=("count", "mean"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .sort_values(["dataset", "intent"])
    )
    intent_summary.to_csv(PER_INTENT_OUTPUT_PATH, index=False)

    # Compare with rules baseline
    rules_intents = compare_with_rules()
    comparison = intent_summary.merge(
        rules_intents[["dataset", "intent", "rules_recall"]],
        on=["dataset", "intent"],
        how="left"
    )
    comparison["if_better"] = comparison["recall_mean"] > comparison["rules_recall"]
    comparison.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print("\n=== Average IF Metrics Across Seeds ===")
    print(summary)

    print("\n=== IF Per-Intent Recall Summary ===")
    print(intent_summary)

    print("\n=== IF vs Rules Per-Intent Comparison ===")
    print(comparison)

    print("\nDone.")
    print(f"Saved: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved: {PER_INTENT_OUTPUT_PATH}")
    print(f"Saved: {COMPARISON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()