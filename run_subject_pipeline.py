from __future__ import annotations

import pandas as pd

from config import DEFAULT_CONFIG
from subject_data import (
    create_user_profiles,
    create_file_catalog,
    generate_normal_logs,
    inject_attack_scenarios,
)
from subject_features import fit_user_baselines, build_subject_features
from subject_rules_eval import split_by_time, apply_rule_baseline, summarize_rule_metrics


def main() -> None:
    cfg = DEFAULT_CONFIG

    print("1) Creating users and file catalog...")
    users = create_user_profiles(cfg["user_counts"], seed=cfg["seed"])
    file_catalog = create_file_catalog(
        users=users,
        files_per_department=cfg["files_per_department"],
        seed=cfg["seed"],
    )

    print("2) Generating normal logs...")
    normal_logs = generate_normal_logs(
        users=users,
        file_catalog=file_catalog,
        days=cfg["days"],
        seed=cfg["seed"],
    )
    normal_logs.to_csv(cfg["paths"]["normal_logs_csv"], index=False)

    print("3) Splitting NORMAL logs by time...")
    normal_logs["timestamp"] = pd.to_datetime(normal_logs["timestamp"])
    train_normal, val_normal, test_normal = split_by_time(
        normal_logs,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
    )

    print("4) Injecting attacks into validation set...")
    val_logs = inject_attack_scenarios(
        logs=val_normal,
        users=users,
        file_catalog=file_catalog,
        attack_counts=cfg["attack_counts_val"],
        seed=cfg["seed"] + 100,
    )

    print("5) Injecting attacks into test set...")
    test_logs = inject_attack_scenarios(
        logs=test_normal,
        users=users,
        file_catalog=file_catalog,
        attack_counts=cfg["attack_counts_test"],
        seed=cfg["seed"] + 200,
    )

    val_logs.to_csv(cfg["paths"]["val_logs_with_attacks_csv"], index=False)
    test_logs.to_csv(cfg["paths"]["test_logs_with_attacks_csv"], index=False)

    # Optional combined export for inspection
    all_logs = pd.concat([train_normal, val_logs, test_logs], ignore_index=True)
    all_logs = all_logs.sort_values("timestamp").reset_index(drop=True)
    all_logs.to_csv(cfg["paths"]["all_logs_with_attacks_csv"], index=False)

    print("6) Fitting user baselines on normal training data...")
    baselines = fit_user_baselines(train_normal)

    print("7) Building features...")
    train_features = build_subject_features(
        train_normal,
        baselines=baselines,
        short_minutes=cfg["windows"]["short_minutes"],
        long_minutes=cfg["windows"]["long_minutes"],
    )

    val_features = build_subject_features(
        val_logs,
        baselines=baselines,
        short_minutes=cfg["windows"]["short_minutes"],
        long_minutes=cfg["windows"]["long_minutes"],
    )

    test_features = build_subject_features(
        test_logs,
        baselines=baselines,
        short_minutes=cfg["windows"]["short_minutes"],
        long_minutes=cfg["windows"]["long_minutes"],
    )

    train_features.to_csv(cfg["paths"]["train_features_csv"], index=False)
    val_features.to_csv(cfg["paths"]["val_features_csv"], index=False)
    test_features.to_csv(cfg["paths"]["test_features_csv"], index=False)

    print("8) Running rules-only baseline...")
    val_rule = apply_rule_baseline(val_features, cfg["rule_thresholds"])
    test_rule = apply_rule_baseline(test_features, cfg["rule_thresholds"])

    val_rule.to_csv(cfg["paths"]["val_rule_results_csv"], index=False)
    test_rule.to_csv(cfg["paths"]["test_rule_results_csv"], index=False)

    print("9) Summarizing metrics...")
    val_metrics = summarize_rule_metrics(val_rule)
    test_metrics = summarize_rule_metrics(test_rule)

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\nSubject-side pipeline finished successfully.")
    print("You now have:")
    print("- normal logs")
    print("- validation logs with attacks")
    print("- test logs with attacks")
    print("- engineered features")
    print("- rules baseline results")
    print("- evaluation metrics")


if __name__ == "__main__":
    main()