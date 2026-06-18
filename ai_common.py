from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


MODEL_FEATURES = [
    "events_10m",
    "views_10m",
    "downloads_10m",
    "edits_10m",
    "deletes_10m",
    "renames_10m",
    "bytes_10m",
    "files_touched_10m",
    "shares_30m",
    "external_shares_30m",
    "permission_changes_30m",
    "new_device",
    "new_city",
    "off_hours",
    "is_weekend",
    "bytes_zscore",
]


def load_feature_sets(
    train_path: str,
    val_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, parse_dates=["timestamp"])
    val_df = pd.read_csv(val_path, parse_dates=["timestamp"])
    test_df = pd.read_csv(test_path, parse_dates=["timestamp"])
    return train_df, val_df, test_df


def prepare_model_inputs(df: pd.DataFrame) -> np.ndarray:
    return df[MODEL_FEATURES].astype(float).values


def select_threshold_by_target_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.85,
) -> float:
    """
    Higher score = more anomalous.
    Choose the highest threshold that still achieves at least target recall.
    """
    unique_thresholds = np.unique(scores)
    unique_thresholds = np.sort(unique_thresholds)

    best_threshold = unique_thresholds.min()
    found = False

    for thr in unique_thresholds:
        y_pred = (scores >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall:
            best_threshold = thr
            found = True

    if not found:
        # fallback: use lowest threshold, which alerts on everything suspicious
        best_threshold = unique_thresholds.min()

    return float(best_threshold)


def alerts_per_1000(alert_series: np.ndarray) -> float:
    if len(alert_series) == 0:
        return 0.0
    return float(np.sum(alert_series)) / float(len(alert_series)) * 1000.0


def mean_time_to_detect(df: pd.DataFrame, alert_col: str) -> float:
    attack_df = df[df["label"] == 1].copy()
    if attack_df.empty:
        return float("nan")

    delays = []
    for scenario_id, g in attack_df.groupby("scenario_id"):
        g = g.sort_values("timestamp")
        first_attack_time = g["timestamp"].min()
        alerted = g[g[alert_col] == 1]
        if not alerted.empty:
            first_alert_time = alerted["timestamp"].min()
            delays.append((first_alert_time - first_attack_time).total_seconds())

    if len(delays) == 0:
        return float("nan")

    return float(np.mean(delays))


def summarize_ai_metrics(
    df: pd.DataFrame,
    score_col: str,
    alert_col: str,
) -> Dict[str, float]:
    y_true = df["label"].astype(int).values
    y_score = df[score_col].astype(float).values
    y_pred = df[alert_col].astype(int).values

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "alerts_per_1000": alerts_per_1000(y_pred),
        "mean_time_to_detect_sec": mean_time_to_detect(df, alert_col),
    }

def per_intent_recall(
    df: pd.DataFrame,
    alert_col: str,
) -> pd.DataFrame:
    """
    Return recall by attack intent.
    Only evaluates rows where label == 1.
    """
    attack_df = df[df["label"] == 1].copy()

    rows = []
    for intent, g in attack_df.groupby("intent"):
        y_true = g["label"].astype(int).values
        y_pred = g[alert_col].astype(int).values

        recall = recall_score(y_true, y_pred, zero_division=0)
        rows.append({
            "intent": intent,
            "count": len(g),
            "recall": float(recall),
        })

    return pd.DataFrame(rows).sort_values("intent").reset_index(drop=True)