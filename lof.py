from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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

VAL_OUTPUT_PATH = "val_lof_results.csv"
TEST_OUTPUT_PATH = "test_lof_results.csv"

TARGET_RECALL = 0.85


def fit_lof(
    X_train: np.ndarray,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    """
    novelty=True allows LOF to score new data after fitting on training data.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        metric="minkowski",
        p=2,  # Euclidean distance
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    """
    sklearn LOF:
    - decision_function: higher = more normal
    We flip sign so that higher = more anomalous
    """
    raw = model.decision_function(X)
    scores = -raw
    return scores


def attach_lof_outputs(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["lof_score"] = scores
    out["lof_alert"] = (out["lof_score"] >= threshold).astype(int)
    return out


def run_single_lof(
    n_neighbors: int = 20,
    target_recall: float = TARGET_RECALL,
) -> dict:
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

    model = fit_lof(X_train_scaled, n_neighbors=n_neighbors)

    val_scores = anomaly_scores(model, X_val_scaled)
    test_scores = anomaly_scores(model, X_test_scaled)

    threshold = select_threshold_by_target_recall(
        y_true=val_df["label"].astype(int).values,
        scores=val_scores,
        target_recall=target_recall,
    )

    val_results = attach_lof_outputs(val_df, val_scores, threshold)
    test_results = attach_lof_outputs(test_df, test_scores, threshold)

    val_metrics = summarize_ai_metrics(val_results, score_col="lof_score", alert_col="lof_alert")
    test_metrics = summarize_ai_metrics(test_results, score_col="lof_score", alert_col="lof_alert")

    val_intent = per_intent_recall(val_results, alert_col="lof_alert")
    val_intent["dataset"] = "val"

    test_intent = per_intent_recall(test_results, alert_col="lof_alert")
    test_intent["dataset"] = "test"

    return {
        "n_neighbors": n_neighbors,
        "threshold": threshold,
        "val_results": val_results,
        "test_results": test_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_intent": val_intent,
        "test_intent": test_intent,
    }


def main() -> None:
    result = run_single_lof(n_neighbors=20, target_recall=TARGET_RECALL)

    print(f"Selected threshold: {result['threshold']:.6f}")

    print("\nValidation LOF metrics:")
    for k, v in result["val_metrics"].items():
        print(f"  {k}: {v}")

    print("\nTest LOF metrics:")
    for k, v in result["test_metrics"].items():
        print(f"  {k}: {v}")

    result["val_results"].to_csv(VAL_OUTPUT_PATH, index=False)
    result["test_results"].to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(f"- {VAL_OUTPUT_PATH}")
    print(f"- {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()