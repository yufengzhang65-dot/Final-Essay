from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ai_common import (
    MODEL_FEATURES,
    load_feature_sets,
    prepare_model_inputs,
    select_threshold_by_target_recall,
    summarize_ai_metrics,
)


TRAIN_FEATURES_PATH = "train_normal_features.csv"
VAL_FEATURES_PATH = "val_features.csv"
TEST_FEATURES_PATH = "test_features.csv"

VAL_OUTPUT_PATH = "val_if_results.csv"
TEST_OUTPUT_PATH = "test_if_results.csv"

TARGET_RECALL = 0.85
RANDOM_SEED = 42


def fit_isolation_forest(
    X_train: np.ndarray,
    random_seed: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    sklearn IsolationForest:
    - decision_function: higher = more normal
    We flip the sign so that higher = more anomalous
    """
    raw = model.decision_function(X)
    scores = -raw
    return scores


def attach_if_outputs(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["if_score"] = scores
    out["if_alert"] = (out["if_score"] >= threshold).astype(int)
    return out


def main() -> None:
    print("1) Loading feature sets...")
    train_df, val_df, test_df = load_feature_sets(
        TRAIN_FEATURES_PATH,
        VAL_FEATURES_PATH,
        TEST_FEATURES_PATH,
    )

    print("2) Preparing model inputs...")
    X_train = prepare_model_inputs(train_df)
    X_val = prepare_model_inputs(val_df)
    X_test = prepare_model_inputs(test_df)

    # Optional but recommended for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("3) Training Isolation Forest on normal training data...")
    model = fit_isolation_forest(X_train_scaled, random_seed=RANDOM_SEED)

    print("4) Scoring validation and test sets...")
    val_scores = anomaly_scores(model, X_val_scaled)
    test_scores = anomaly_scores(model, X_test_scaled)

    print(f"5) Selecting threshold on validation set (target recall = {TARGET_RECALL})...")
    y_val = val_df["label"].astype(int).values
    threshold = select_threshold_by_target_recall(
        y_true=y_val,
        scores=val_scores,
        target_recall=TARGET_RECALL,
    )
    print(f"Selected threshold: {threshold:.6f}")

    print("6) Attaching predictions...")
    val_results = attach_if_outputs(val_df, val_scores, threshold)
    test_results = attach_if_outputs(test_df, test_scores, threshold)

    print("7) Saving outputs...")
    val_results.to_csv(VAL_OUTPUT_PATH, index=False)
    test_results.to_csv(TEST_OUTPUT_PATH, index=False)

    print("8) Summarizing metrics...")
    val_metrics = summarize_ai_metrics(val_results, score_col="if_score", alert_col="if_alert")
    test_metrics = summarize_ai_metrics(test_results, score_col="if_score", alert_col="if_alert")

    print("\nValidation IF metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print("\nTest IF metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\nIsolation Forest pipeline finished successfully.")
    print("Outputs:")
    print(f"- {VAL_OUTPUT_PATH}")
    print(f"- {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()