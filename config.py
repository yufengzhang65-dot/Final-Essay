from __future__ import annotations

DEFAULT_CONFIG = {
    "seed": 42,
    "days": 30,
    "user_counts": {
        "office_staff": 20,
        "manager": 5,
        "contractor": 3,
        "it_admin": 2,
    },
    "files_per_department": 120,
    "departments": ["operations", "management", "external", "it"],
    "split": {
        "train_frac": 0.6,
        "val_frac": 0.2,
    },
    "windows": {
        "short_minutes": 10,
        "long_minutes": 30,
    },
    # NEW: inject attacks separately into val and test
    "attack_counts_val": {
        "exfiltration": 2,
        "recon": 2,
        "privilege_misuse": 2,
        "tamper_like": 2,
    },
    "attack_counts_test": {
        "exfiltration": 2,
        "recon": 2,
        "privilege_misuse": 2,
        "tamper_like": 2,
    },
    "rule_thresholds": {
        "exfil_downloads_10m": 15,
        "exfil_bytes_10m": 10_000_000,
        "recon_views_10m": 18,
        "recon_files_10m": 14,
        "privilege_changes_30m": 5,
        "tamper_deletes_renames_10m": 5,
        "tamper_edits_10m": 10,
    },
    "paths": {
        "normal_logs_csv": "normal_logs.csv",
        "all_logs_with_attacks_csv": "all_logs_with_attacks.csv",
        "val_logs_with_attacks_csv": "val_logs_with_attacks.csv",
        "test_logs_with_attacks_csv": "test_logs_with_attacks.csv",
        "train_features_csv": "train_normal_features.csv",
        "val_features_csv": "val_features.csv",
        "test_features_csv": "test_features.csv",
        "val_rule_results_csv": "val_rule_results.csv",
        "test_rule_results_csv": "test_rule_results.csv",
    },
}