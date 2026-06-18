from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent

overall_file = BASE_DIR / "table_07_overall_test_results.csv"
per_intent_file = BASE_DIR / "table_08_per_intent_recall.csv"

if not overall_file.exists():
    raise FileNotFoundError("Cannot find table_07_overall_test_results.csv")

if not per_intent_file.exists():
    raise FileNotFoundError("Cannot find table_08_per_intent_recall.csv")

overall_df = pd.read_csv(overall_file)
per_intent_df = pd.read_csv(per_intent_file)

# =========================
# Common settings
# =========================
model_order = ["Rules", "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Autoencoder"]

model_short = {
    "Rules": "Rules",
    "Isolation Forest": "IF",
    "One-Class SVM": "OCSVM",
    "Local Outlier Factor": "LOF",
    "Autoencoder": "AE"
}

# =========================
# Figure 5 (fixed):
# Detection Coverage vs Alert Burden Trade-Off
# =========================
def make_fixed_figure5():
    df = overall_df.copy()

    # Ensure correct model order
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)
    df["Short"] = df["Model"].map(model_short)

    x = df["Alerts per 1,000"]
    y = df["Recall"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x, y, s=120)

    # Manual label offsets to avoid overlap
    label_offsets = {
        "Rules": (8, 8),
        "IF": (8, 8),
        "OCSVM": (10, 10),
        "AE": (10, -14),
        "LOF": (10, 8)
    }

    for _, row in df.iterrows():
        short_name = row["Short"]
        dx, dy = label_offsets.get(short_name, (8, 8))
        ax.annotate(
            short_name,
            (row["Alerts per 1,000"], row["Recall"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10
        )

    ax.set_xlabel("Alerts per 1,000 events", fontsize=12)
    ax.set_ylabel("Test recall", fontsize=12)
    ax.set_title("Detection Coverage vs Alert Burden Trade-Off", fontsize=15)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_05_recall_alert_tradeoff_fixed.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 6 (fixed):
# Per-Attack Recall by Detection Method
# =========================
def make_fixed_figure6():
    df = per_intent_df.copy()

    # Ensure correct model order
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)

    # Desired attack order
    attack_order = [
        "Exfiltration",
        "Reconnaissance",
        "Privilege misuse",
        "Tamper-like"
    ]

    # Short model names
    df["ModelShort"] = df["Model"].map(model_short)

    # Set up plotting values
    x = np.arange(len(attack_order))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[col] for col in attack_order]
        ax.bar(
            x + (i - 2) * width,
            values,
            width,
            label=row["ModelShort"]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(attack_order, rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Per-Attack Recall by Detection Method", fontsize=15)
    ax.legend(ncol=5, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_06_per_intent_recall_fixed.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    make_fixed_figure5()
    make_fixed_figure6()
    print("Done.")
    print("Generated:")
    print(" - fig_05_recall_alert_tradeoff_fixed.png")
    print(" - fig_06_per_intent_recall_fixed.png")