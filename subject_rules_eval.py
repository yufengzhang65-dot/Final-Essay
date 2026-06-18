from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def split_by_time(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def apply_rule_baseline(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()

    scores: List[float] = []
    alerts: List[int] = []
    predicted_intents: List[str] = []
    reasons: List[str] = []

    for _, row in out.iterrows():
        intent_scores = {
            "exfiltration": 0.0,
            "recon": 0.0,
            "privilege_misuse": 0.0,
            "tamper_like": 0.0,
        }
        intent_reasons = {
            "exfiltration": [],
            "recon": [],
            "privilege_misuse": [],
            "tamper_like": [],
        }

        # -------- Exfiltration --------
        if row["downloads_10m"] >= thresholds["exfil_downloads_10m"]:
            intent_scores["exfiltration"] += 1.5
            intent_reasons["exfiltration"].append("high_download_burst")

        if row["bytes_10m"] >= thresholds["exfil_bytes_10m"]:
            intent_scores["exfiltration"] += 1.0
            intent_reasons["exfiltration"].append("high_bytes_10m")

        # only strong if current event is share/download related
        if row["external_shares_30m"] >= 1 and row["action"] in {"share", "download"}:
            intent_scores["exfiltration"] += 1.0
            intent_reasons["exfiltration"].append("external_share_context")

        if row["new_device"] == 1:
            intent_scores["exfiltration"] += 0.5
            intent_reasons["exfiltration"].append("new_device")

        if row["off_hours"] == 1:
            intent_scores["exfiltration"] += 0.5
            intent_reasons["exfiltration"].append("off_hours")

        # -------- Recon --------
        if row["views_10m"] >= thresholds["recon_views_10m"]:
            intent_scores["recon"] += 1.2
            intent_reasons["recon"].append("many_views")

        if row["files_touched_10m"] >= thresholds["recon_files_10m"]:
            intent_scores["recon"] += 1.2
            intent_reasons["recon"].append("many_files")

        # only for view-heavy behaviour
        if (
            row["action"] == "view"
            and row["bytes_10m"] < 500_000
            and row["edits_10m"] == 0
            and row["downloads_10m"] <= 2
        ):
            intent_scores["recon"] += 0.7
            intent_reasons["recon"].append("low_bytes_probe_pattern")

        # -------- Privilege misuse --------
        if row["permission_changes_30m"] >= thresholds["privilege_changes_30m"]:
            intent_scores["privilege_misuse"] += 1.5
            intent_reasons["privilege_misuse"].append("permission_spike")

        if row["permission_change"] == 1 and row["new_device"] == 1:
            intent_scores["privilege_misuse"] += 0.8
            intent_reasons["privilege_misuse"].append("permission_change_new_device")

        # -------- Tamper-like --------
        if (row["deletes_10m"] + row["renames_10m"]) >= thresholds["tamper_deletes_renames_10m"]:
            intent_scores["tamper_like"] += 1.5
            intent_reasons["tamper_like"].append("delete_rename_burst")

        if row["edits_10m"] >= thresholds["tamper_edits_10m"]:
            intent_scores["tamper_like"] += 1.0
            intent_reasons["tamper_like"].append("edit_burst")

        if row["off_hours"] == 1 and (row["deletes_10m"] + row["renames_10m"]) > 0:
            intent_scores["tamper_like"] += 0.5
            intent_reasons["tamper_like"].append("tamper_off_hours")

        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]

        is_alert = 1 if best_score >= 1.5 else 0
        predicted_intent = best_intent if is_alert else "normal"
        best_reasons = "|".join(intent_reasons[best_intent]) if is_alert else ""

        scores.append(best_score)
        alerts.append(is_alert)
        predicted_intents.append(predicted_intent)
        reasons.append(best_reasons)

    out["rule_score"] = scores
    out["rule_alert"] = alerts
    out["predicted_intent"] = predicted_intents
    out["rule_reasons"] = reasons
    return out


def alerts_per_1000(alert_series: pd.Series) -> float:
    if len(alert_series) == 0:
        return 0.0
    return float(alert_series.sum()) / float(len(alert_series)) * 1000.0


def time_to_detect(df: pd.DataFrame) -> float:
    attack_df = df[df["label"] == 1].copy()
    if attack_df.empty:
        return np.nan

    delays = []
    for scenario_id, g in attack_df.groupby("scenario_id"):
        g = g.sort_values("timestamp")
        first_attack_time = g["timestamp"].min()
        alerted = g[g["rule_alert"] == 1]
        if not alerted.empty:
            first_alert_time = alerted["timestamp"].min()
            delay_sec = (first_alert_time - first_attack_time).total_seconds()
            delays.append(delay_sec)

    if not delays:
        return np.nan
    return float(np.mean(delays))


def summarize_rule_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["label"].astype(int).values
    y_pred = df["rule_alert"].astype(int).values
    y_score = df["rule_score"].astype(float).values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "alerts_per_1000": alerts_per_1000(df["rule_alert"]),
        "mean_time_to_detect_sec": time_to_detect(df),
    }