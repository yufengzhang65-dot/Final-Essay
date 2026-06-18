
"""
make_paper_outputs_clean_names.py

Generate dissertation tables and figures using ONLY clean CSV filenames.
This version does NOT look for files with suffixes such as (1) or (2).

Put this script in the same folder as your CSV files, then run:

    python make_paper_outputs_clean_names.py

Expected CSV files in the same folder:
- test_rule_results.csv
- if_experiment_summary.csv
- if_per_intent_summary.csv
- ocsvm_experiment_summary.csv
- ocsvm_per_intent_summary.csv
- lof_experiment_summary.csv
- lof_per_intent_summary.csv
- ae_experiment_summary.csv
- ae_per_intent_summary.csv

Outputs:
paper_outputs/
├── tables/
└── figures/
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================
# Paths and settings
# =========================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "paper_outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "Rules",
    "Isolation Forest",
    "One-Class SVM",
    "Local Outlier Factor",
    "Autoencoder"
]

INTENT_ORDER = [
    "exfiltration",
    "recon",
    "privilege_misuse",
    "tamper_like"
]

INTENT_LABELS = {
    "exfiltration": "Exfiltration",
    "recon": "Reconnaissance",
    "privilege_misuse": "Privilege misuse",
    "tamper_like": "Tamper-like"
}

MODEL_SHORT = {
    "Rules": "Rules",
    "Isolation Forest": "IF",
    "One-Class SVM": "OCSVM",
    "Local Outlier Factor": "LOF",
    "Autoencoder": "AE"
}


# =========================
# File loading helpers
# =========================

def require_csv(filename: str) -> Path:
    """
    Load only the exact clean filename.
    This function intentionally does not search for files with (1), (2), etc.
    """
    path = BASE_DIR / filename
    if not path.exists():
        existing_csvs = sorted(p.name for p in BASE_DIR.glob("*.csv"))
        existing_text = "\n".join(f"  - {name}" for name in existing_csvs) if existing_csvs else "  No CSV files found."
        raise FileNotFoundError(
            f"\nRequired file not found: {filename}\n\n"
            f"This clean-name version only looks for exact filenames.\n"
            f"Please make sure the file is in the same folder as this script.\n\n"
            f"CSV files currently found in this folder:\n{existing_text}\n"
        )
    return path


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(require_csv(filename))


def safe_round(value, digits=3):
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def normalise_gamma_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def summary_metric(summary_df, metric_name, column="test_mean"):
    row = summary_df[summary_df["metric"] == metric_name]
    if len(row) == 0:
        return np.nan
    return float(row.iloc[0][column])


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Model selection helpers
# =========================

def select_ocsvm_row(df):
    """
    Select the dissertation's main OCSVM configuration:
    nu = 0.05, gamma = 0.05.

    If that exact config is missing, fall back to the best validation F1,
    then test F1.
    """
    gamma_text = df["gamma"].apply(normalise_gamma_value)
    preferred = df[
        (np.isclose(df["nu"].astype(float), 0.05))
        & (gamma_text == "0.05")
    ]

    if len(preferred) > 0:
        return preferred.iloc[0]

    warnings.warn(
        "Preferred OCSVM config nu=0.05, gamma=0.05 not found. "
        "Falling back to best available configuration."
    )
    if "val_f1" in df.columns:
        return df.sort_values("val_f1", ascending=False).iloc[0]
    return df.sort_values("test_f1", ascending=False).iloc[0]


def select_lof_row(df):
    """
    Select the dissertation's main LOF configuration:
    n_neighbors = 15.

    If that exact config is missing, fall back to the best validation F1,
    then test F1.
    """
    preferred = df[df["n_neighbors"].astype(int) == 15]

    if len(preferred) > 0:
        return preferred.iloc[0]

    warnings.warn(
        "Preferred LOF config n_neighbors=15 not found. "
        "Falling back to best available configuration."
    )
    if "val_f1" in df.columns:
        return df.sort_values("val_f1", ascending=False).iloc[0]
    return df.sort_values("test_f1", ascending=False).iloc[0]


# =========================
# Metric calculation
# =========================

def calculate_time_to_detect(df, alert_col):
    """
    Calculate mean time-to-detect in seconds.

    For each attack scenario:
    first attack timestamp -> first detected attack timestamp in the same scenario.

    Scenarios with no detected attack event are ignored in the mean.
    """
    if "timestamp" not in df.columns or "scenario_id" not in df.columns:
        return np.nan

    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")

    attacks = tmp[(tmp["label"].astype(int) == 1) & tmp["scenario_id"].notna()].copy()
    if attacks.empty:
        return np.nan

    delays = []
    for _, group in attacks.groupby("scenario_id"):
        group = group.sort_values("timestamp")
        start_time = group["timestamp"].min()
        detected = group[group[alert_col].astype(int) == 1]

        if len(detected) > 0:
            first_detect = detected["timestamp"].min()
            delays.append((first_detect - start_time).total_seconds())

    if len(delays) == 0:
        return np.nan

    return float(np.mean(delays))


def calculate_rules_overall():
    df = load_csv("test_rule_results.csv")

    required_cols = {"label", "rule_alert", "rule_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"test_rule_results.csv is missing columns: {sorted(missing)}")

    y_true = df["label"].astype(int)
    y_pred = df["rule_alert"].astype(int)

    if SKLEARN_AVAILABLE:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, df["rule_score"])
    else:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        pr_auc = np.nan

    alerts_per_1000 = y_pred.mean() * 1000
    time_to_detect = calculate_time_to_detect(df, "rule_alert")

    return {
        "Model": "Rules",
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "Alerts per 1,000": alerts_per_1000,
        "Mean time-to-detect (s)": time_to_detect
    }


def calculate_rules_per_intent():
    df = load_csv("test_rule_results.csv")

    required_cols = {"label", "intent", "rule_alert"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"test_rule_results.csv is missing columns: {sorted(missing)}")

    attacks = df[df["label"].astype(int) == 1].copy()

    rows = []
    for intent in INTENT_ORDER:
        sub = attacks[attacks["intent"] == intent]
        recall = sub["rule_alert"].astype(int).mean() if len(sub) else np.nan
        rows.append({
            "Model": "Rules",
            "Intent": intent,
            "Recall": recall
        })

    return pd.DataFrame(rows)


# =========================
# Build result tables
# =========================

def build_overall_results_table():
    rows = []

    # Rules baseline
    rows.append(calculate_rules_overall())

    # Isolation Forest
    if_summary = load_csv("if_experiment_summary.csv")
    rows.append({
        "Model": "Isolation Forest",
        "Precision": summary_metric(if_summary, "precision"),
        "Recall": summary_metric(if_summary, "recall"),
        "F1": summary_metric(if_summary, "f1"),
        "PR-AUC": summary_metric(if_summary, "pr_auc"),
        "Alerts per 1,000": summary_metric(if_summary, "alerts_per_1000"),
        "Mean time-to-detect (s)": summary_metric(if_summary, "mean_time_to_detect_sec")
    })

    # One-Class SVM
    ocsvm_summary = load_csv("ocsvm_experiment_summary.csv")
    ocsvm_row = select_ocsvm_row(ocsvm_summary)
    rows.append({
        "Model": "One-Class SVM",
        "Precision": ocsvm_row["test_precision"],
        "Recall": ocsvm_row["test_recall"],
        "F1": ocsvm_row["test_f1"],
        "PR-AUC": ocsvm_row["test_pr_auc"],
        "Alerts per 1,000": ocsvm_row["test_alerts_per_1000"],
        "Mean time-to-detect (s)": ocsvm_row["test_time_to_detect"]
    })

    # Local Outlier Factor
    lof_summary = load_csv("lof_experiment_summary.csv")
    lof_row = select_lof_row(lof_summary)
    rows.append({
        "Model": "Local Outlier Factor",
        "Precision": lof_row["test_precision"],
        "Recall": lof_row["test_recall"],
        "F1": lof_row["test_f1"],
        "PR-AUC": lof_row["test_pr_auc"],
        "Alerts per 1,000": lof_row["test_alerts_per_1000"],
        "Mean time-to-detect (s)": lof_row["test_time_to_detect"]
    })

    # Autoencoder
    ae_summary = load_csv("ae_experiment_summary.csv")
    rows.append({
        "Model": "Autoencoder",
        "Precision": summary_metric(ae_summary, "precision"),
        "Recall": summary_metric(ae_summary, "recall"),
        "F1": summary_metric(ae_summary, "f1"),
        "PR-AUC": summary_metric(ae_summary, "pr_auc"),
        "Alerts per 1,000": summary_metric(ae_summary, "alerts_per_1000"),
        "Mean time-to-detect (s)": summary_metric(ae_summary, "mean_time_to_detect_sec")
    })

    table = pd.DataFrame(rows)
    table["Model"] = pd.Categorical(table["Model"], categories=MODEL_ORDER, ordered=True)
    table = table.sort_values("Model").reset_index(drop=True)
    table["Model"] = table["Model"].astype(str)

    rounded = table.copy()
    for col in ["Precision", "Recall", "F1", "PR-AUC"]:
        rounded[col] = rounded[col].map(lambda x: safe_round(x, 3))
    for col in ["Alerts per 1,000", "Mean time-to-detect (s)"]:
        rounded[col] = rounded[col].map(lambda x: safe_round(x, 2))

    return table, rounded


def build_per_intent_table():
    frames = []

    # Rules
    frames.append(calculate_rules_per_intent())

    # Isolation Forest
    if_intent = load_csv("if_per_intent_summary.csv")
    if_test = if_intent[if_intent["dataset"] == "test"].copy()
    frames.append(pd.DataFrame({
        "Model": "Isolation Forest",
        "Intent": if_test["intent"],
        "Recall": if_test["recall_mean"]
    }))

    # One-Class SVM
    ocsvm_summary = load_csv("ocsvm_experiment_summary.csv")
    ocsvm_row = select_ocsvm_row(ocsvm_summary)

    ocsvm_intent = load_csv("ocsvm_per_intent_summary.csv")
    gamma_text = ocsvm_intent["gamma"].apply(normalise_gamma_value)
    selected_gamma = normalise_gamma_value(ocsvm_row["gamma"])

    ocsvm_test = ocsvm_intent[
        (ocsvm_intent["dataset"] == "test")
        & (np.isclose(ocsvm_intent["nu"].astype(float), float(ocsvm_row["nu"])))
        & (gamma_text == selected_gamma)
    ].copy()

    frames.append(pd.DataFrame({
        "Model": "One-Class SVM",
        "Intent": ocsvm_test["intent"],
        "Recall": ocsvm_test["recall"]
    }))

    # Local Outlier Factor
    lof_summary = load_csv("lof_experiment_summary.csv")
    lof_row = select_lof_row(lof_summary)

    lof_intent = load_csv("lof_per_intent_summary.csv")
    lof_test = lof_intent[
        (lof_intent["dataset"] == "test")
        & (lof_intent["n_neighbors"].astype(int) == int(lof_row["n_neighbors"]))
    ].copy()

    frames.append(pd.DataFrame({
        "Model": "Local Outlier Factor",
        "Intent": lof_test["intent"],
        "Recall": lof_test["recall"]
    }))

    # Autoencoder
    ae_intent = load_csv("ae_per_intent_summary.csv")
    ae_test = ae_intent[ae_intent["dataset"] == "test"].copy()
    frames.append(pd.DataFrame({
        "Model": "Autoencoder",
        "Intent": ae_test["intent"],
        "Recall": ae_test["recall_mean"]
    }))

    long_table = pd.concat(frames, ignore_index=True)

    long_table["Model"] = pd.Categorical(long_table["Model"], categories=MODEL_ORDER, ordered=True)
    long_table["Intent"] = pd.Categorical(long_table["Intent"], categories=INTENT_ORDER, ordered=True)
    long_table = long_table.sort_values(["Model", "Intent"]).reset_index(drop=True)
    long_table["Model"] = long_table["Model"].astype(str)
    long_table["Intent"] = long_table["Intent"].astype(str)

    wide = long_table.pivot(index="Model", columns="Intent", values="Recall").reset_index()
    wide["Model"] = pd.Categorical(wide["Model"], categories=MODEL_ORDER, ordered=True)
    wide = wide.sort_values("Model")
    wide["Model"] = wide["Model"].astype(str)
    wide = wide[["Model"] + INTENT_ORDER]

    rounded = wide.copy()
    for col in INTENT_ORDER:
        rounded[col] = rounded[col].map(lambda x: safe_round(x, 3))

    rounded = rounded.rename(columns=INTENT_LABELS)

    return long_table, rounded


# =========================
# Static paper tables
# =========================

def write_static_tables():
    research_objectives = pd.DataFrame([
        {
            "Item": "Research question",
            "Content": "Which lightweight anomaly-detection approach can best reduce alert burden while maintaining strong detection coverage on cloud file-access audit logs in SME settings?"
        },
        {
            "Item": "Objective 1",
            "Content": "Build a controlled SME-style cloud audit-log pipeline."
        },
        {
            "Item": "Objective 2",
            "Content": "Compare a rules-only baseline with lightweight machine-learning methods and a simple deep-learning comparator."
        },
        {
            "Item": "Objective 3",
            "Content": "Evaluate detection coverage together with operational alert burden."
        }
    ])
    research_objectives.to_csv(TABLE_DIR / "table_01_research_question_and_objectives.csv", index=False)

    personas = pd.DataFrame([
        {
            "Persona": "Office staff",
            "Normal behaviour": "Routine file access, edits, downloads, mostly during standard work hours.",
            "Why included": "Represents the main everyday business user baseline."
        },
        {
            "Persona": "Manager",
            "Normal behaviour": "Broader file access and more sharing behaviour than ordinary staff.",
            "Why included": "Represents higher access scope and potentially sensitive sharing contexts."
        },
        {
            "Persona": "Contractor",
            "Normal behaviour": "Narrower project-based access with more variable context.",
            "Why included": "Represents temporary or external users with less stable patterns."
        },
        {
            "Persona": "IT administrator",
            "Normal behaviour": "More permission-related and administrative actions.",
            "Why included": "Provides normal context for permission activity and privilege-misuse comparisons."
        }
    ])
    personas.to_csv(TABLE_DIR / "table_02_user_personas.csv", index=False)

    attacks = pd.DataFrame([
        {
            "Attack category": "Exfiltration",
            "Behavioural meaning": "Data leakage or removal through access and sharing activity.",
            "Example signals": "Burst downloads, high byte volume, external/public sharing."
        },
        {
            "Attack category": "Reconnaissance",
            "Behavioural meaning": "Broad exploration of available files before stronger action.",
            "Example signals": "Many views, many files touched, low data volume."
        },
        {
            "Attack category": "Privilege misuse",
            "Behavioural meaning": "Abnormal or abusive access-control changes.",
            "Example signals": "Repeated permission changes, unusual device or context."
        },
        {
            "Attack category": "Tamper-like",
            "Behavioural meaning": "Suspicious modification, deletion, or disruption of files.",
            "Example signals": "Bursts of rename, edit, or delete activity."
        }
    ])
    attacks.to_csv(TABLE_DIR / "table_03_attack_scenario_design.csv", index=False)

    features = pd.DataFrame([
        {
            "Feature group": "Short-window activity",
            "Examples": "events_10m, views_10m, downloads_10m, files_touched_10m",
            "Purpose": "Captures bursty access behaviour and short-term activity spikes."
        },
        {
            "Feature group": "Short-window modification",
            "Examples": "edits_10m, deletes_10m, renames_10m",
            "Purpose": "Captures rapid modification or tamper-like patterns."
        },
        {
            "Feature group": "Long-window sharing and permissions",
            "Examples": "shares_30m, external_shares_30m, permission_changes_30m",
            "Purpose": "Captures accumulated sharing and access-control risk."
        },
        {
            "Feature group": "Context features",
            "Examples": "new_device, new_city, off_hours, is_weekend",
            "Purpose": "Captures unusual operational context."
        },
        {
            "Feature group": "User-relative volume",
            "Examples": "bytes_10m, bytes_zscore",
            "Purpose": "Captures high byte volume and deviation from the user's normal baseline."
        }
    ])
    features.to_csv(TABLE_DIR / "table_04_engineered_feature_groups.csv", index=False)

    methods = pd.DataFrame([
        {
            "Method": "Rules baseline",
            "Type": "Transparent heuristic",
            "Role in project": "Practical SME-style starting point with human-readable reasons."
        },
        {
            "Method": "Isolation Forest",
            "Type": "Tree-based anomaly detection",
            "Role in project": "Tests sensitivity to strong, bursty outlier behaviour."
        },
        {
            "Method": "One-Class SVM",
            "Type": "Boundary-based anomaly detection",
            "Role in project": "Tests whether normal-boundary deviation captures subtle suspicious activity."
        },
        {
            "Method": "Local Outlier Factor",
            "Type": "Density-based anomaly detection",
            "Role in project": "Tests sensitivity to locally unusual behaviour."
        },
        {
            "Method": "Autoencoder",
            "Type": "Reconstruction-based deep learning",
            "Role in project": "Tests whether a simple neural comparator adds practical value."
        }
    ])
    methods.to_csv(TABLE_DIR / "table_05_detection_methods_compared.csv", index=False)

    config = pd.DataFrame([
        {
            "Model": "Rules baseline",
            "Main configuration": "Intent-specific rule scoring; alert raised when best rule score exceeds threshold."
        },
        {
            "Model": "Isolation Forest",
            "Main configuration": "Normal-only training; repeated seeds; anomaly score threshold selected on validation set."
        },
        {
            "Model": "One-Class SVM",
            "Main configuration": "RBF kernel; selected configuration nu = 0.05, gamma = 0.05."
        },
        {
            "Model": "Local Outlier Factor",
            "Main configuration": "Novelty detection; selected configuration n_neighbors = 15."
        },
        {
            "Model": "Autoencoder",
            "Main configuration": "16 -> 8 -> 4 -> 8 -> 16 reconstruction model; normal-only training; repeated seeds."
        }
    ])
    config.to_csv(TABLE_DIR / "table_06_model_configuration.csv", index=False)


# =========================
# Figures
# =========================

def draw_pipeline_diagram(steps, title, filename):
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")

    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.5

    for i, (x, step) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            y,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5")
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.055, y),
                xytext=(x + 0.055, y),
                arrowprops=dict(arrowstyle="->")
            )

    ax.set_title(title, fontsize=13, pad=20)
    save_current_figure(filename)


def plot_model_performance(overall_table):
    metrics = ["Precision", "Recall", "F1", "PR-AUC"]
    plot_df = overall_table[["Model"] + metrics].copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)

    x = np.arange(len(plot_df["Model"]))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, plot_df[metric], width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Test-Set Performance by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_current_figure("fig_03_model_performance_comparison.png")


def plot_alert_burden(overall_table):
    plot_df = overall_table[["Model", "Alerts per 1,000"]].copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(plot_df["Model"], plot_df["Alerts per 1,000"])
    ax.set_ylabel("Alerts per 1,000 events")
    ax.set_title("Alert Burden by Detection Method")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(plot_df["Alerts per 1,000"]):
        ax.text(i, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save_current_figure("fig_04_alert_burden_comparison.png")


def plot_recall_alert_tradeoff(overall_table):
    plot_df = overall_table[["Model", "Recall", "Alerts per 1,000"]].copy()
    plot_df["Short"] = plot_df["Model"].map(MODEL_SHORT)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(plot_df["Alerts per 1,000"], plot_df["Recall"], s=90)

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["Short"],
            (row["Alerts per 1,000"], row["Recall"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9
        )

    ax.set_xlabel("Alerts per 1,000 events")
    ax.set_ylabel("Test recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Detection Coverage vs Alert Burden Trade-Off")
    ax.grid(alpha=0.3)

    save_current_figure("fig_05_recall_alert_tradeoff.png")


def plot_per_intent_recall(per_intent_long):
    plot_df = per_intent_long.copy()
    plot_df["ModelShort"] = plot_df["Model"].map(MODEL_SHORT)
    plot_df["IntentLabel"] = plot_df["Intent"].map(INTENT_LABELS)

    pivot = plot_df.pivot(index="IntentLabel", columns="ModelShort", values="Recall")
    pivot = pivot[[MODEL_SHORT[m] for m in MODEL_ORDER]]

    x = np.arange(len(pivot.index))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(pivot.columns):
        ax.bar(x + (i - 2) * width, pivot[model].values, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Attack Recall by Detection Method")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    save_current_figure("fig_06_per_intent_recall_comparison.png")


def plot_time_to_detect(overall_table):
    plot_df = overall_table[["Model", "Mean time-to-detect (s)"]].copy()
    plot_df["Model"] = plot_df["Model"].map(MODEL_SHORT)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(plot_df["Model"], plot_df["Mean time-to-detect (s)"])
    ax.set_ylabel("Mean time-to-detect (seconds)")
    ax.set_title("Mean Time-to-Detect by Detection Method")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(plot_df["Mean time-to-detect (s)"]):
        if not pd.isna(value):
            ax.text(i, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    save_current_figure("fig_07_time_to_detect_comparison.png")


def plot_ocsvm_configuration_sensitivity():
    df = load_csv("ocsvm_experiment_summary.csv").copy()
    df["Config"] = "nu=" + df["nu"].astype(str) + ", gamma=" + df["gamma"].astype(str)
    df = df.sort_values("test_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["Config"], df["test_f1"])
    ax.set_xlabel("Test F1-score")
    ax.set_title("OCSVM Configuration Sensitivity")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)

    save_current_figure("fig_08_ocsvm_configuration_sensitivity.png")


# =========================
# Main
# =========================

def main():
    print("Generating dissertation tables and figures with clean filenames only...")
    print(f"Input folder: {BASE_DIR}")
    print(f"Output folder: {OUT_DIR}")

    write_static_tables()

    overall_raw, overall_rounded = build_overall_results_table()
    per_intent_long, per_intent_rounded = build_per_intent_table()

    overall_rounded.to_csv(TABLE_DIR / "table_07_overall_test_results.csv", index=False)
    per_intent_rounded.to_csv(TABLE_DIR / "table_08_per_intent_recall.csv", index=False)
    per_intent_long.to_csv(TABLE_DIR / "table_08_per_intent_recall_long_format.csv", index=False)

    draw_pipeline_diagram(
        [
            "Synthetic SME\naudit-log generation",
            "Attack scenario\ninjection",
            "Feature\nengineering",
            "Rules baseline\n+ AI models",
            "Validation\nthreshold selection",
            "Test-set\nevaluation",
            "SME suitability\ndiscussion"
        ],
        "Overall Research Pipeline",
        "fig_01_overall_research_pipeline.png"
    )

    draw_pipeline_diagram(
        [
            "User\npersonas",
            "File\ncatalogue",
            "Normal log\ngeneration",
            "Attack\ninjection",
            "Engineered\nfeature table"
        ],
        "Subject-Side Synthetic Audit-Log Pipeline",
        "fig_02_subject_side_pipeline.png"
    )

    plot_model_performance(overall_raw)
    plot_alert_burden(overall_raw)
    plot_recall_alert_tradeoff(overall_raw)
    plot_per_intent_recall(per_intent_long)
    plot_time_to_detect(overall_raw)
    plot_ocsvm_configuration_sensitivity()

    print("\nDone.")

    print("\nGenerated tables:")
    for path in sorted(TABLE_DIR.glob("*.csv")):
        print(f"  - {path.relative_to(BASE_DIR)}")

    print("\nGenerated figures:")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
