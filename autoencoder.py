from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

VAL_OUTPUT_PATH = "val_ae_results.csv"
TEST_OUTPUT_PATH = "test_ae_results.csv"

TARGET_RECALL = 0.85
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AEConfig:
    input_dim: int = 16
    hidden_dim: int = 8
    bottleneck_dim: int = 4
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 8
    train_val_split: float = 0.15


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_internal_validation(
    X_train: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)

    split_idx = int(len(idx) * (1 - val_frac))
    train_idx = idx[:split_idx]
    val_idx = idx[split_idx:]

    return X_train[train_idx], X_train[val_idx]


def make_dataloader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(tensor_x, tensor_x)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    X_train: np.ndarray,
    config: AEConfig,
    seed: int = 42,
) -> Tuple[SimpleAutoencoder, dict]:
    set_global_seed(seed)

    X_fit, X_inner_val = split_train_internal_validation(
        X_train, val_frac=config.train_val_split, seed=seed
    )

    train_loader = make_dataloader(X_fit, config.batch_size, shuffle=True)
    val_loader = make_dataloader(X_inner_val, config.batch_size, shuffle=False)

    model = SimpleAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        bottleneck_dim=config.bottleneck_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validate on normal-only holdout from train
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history["train_loss"]),
        "history": history,
    }
    return model, info


def reconstruction_errors(model: SimpleAutoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor_x = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model(tensor_x)
        per_row_mse = torch.mean((recon - tensor_x) ** 2, dim=1)

    return per_row_mse.cpu().numpy()


def attach_ae_outputs(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["ae_score"] = scores
    out["ae_alert"] = (out["ae_score"] >= threshold).astype(int)
    return out


def run_single_autoencoder(
    config: AEConfig,
    seed: int = 42,
    target_recall: float = TARGET_RECALL,
) -> dict:
    set_global_seed(seed)

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

    model, train_info = train_autoencoder(
        X_train_scaled,
        config=config,
        seed=seed,
    )

    val_scores = reconstruction_errors(model, X_val_scaled)
    test_scores = reconstruction_errors(model, X_test_scaled)

    threshold = select_threshold_by_target_recall(
        y_true=val_df["label"].astype(int).values,
        scores=val_scores,
        target_recall=target_recall,
    )

    val_results = attach_ae_outputs(val_df, val_scores, threshold)
    test_results = attach_ae_outputs(test_df, test_scores, threshold)

    val_metrics = summarize_ai_metrics(val_results, score_col="ae_score", alert_col="ae_alert")
    test_metrics = summarize_ai_metrics(test_results, score_col="ae_score", alert_col="ae_alert")

    val_intent = per_intent_recall(val_results, alert_col="ae_alert")
    val_intent["dataset"] = "val"
    val_intent["seed"] = seed

    test_intent = per_intent_recall(test_results, alert_col="ae_alert")
    test_intent["dataset"] = "test"
    test_intent["seed"] = seed

    return {
        "seed": seed,
        "threshold": threshold,
        "train_info": train_info,
        "val_results": val_results,
        "test_results": test_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_intent": val_intent,
        "test_intent": test_intent,
    }


def main() -> None:
    config = AEConfig()

    result = run_single_autoencoder(
        config=config,
        seed=42,
        target_recall=TARGET_RECALL,
    )

    print(f"Device: {DEVICE}")
    print(f"Selected threshold: {result['threshold']:.6f}")
    print(f"Epochs ran: {result['train_info']['epochs_ran']}")
    print(f"Best inner-val loss: {result['train_info']['best_val_loss']:.6f}")

    print("\nValidation AE metrics:")
    for k, v in result["val_metrics"].items():
        print(f"  {k}: {v}")

    print("\nTest AE metrics:")
    for k, v in result["test_metrics"].items():
        print(f"  {k}: {v}")

    result["val_results"].to_csv(VAL_OUTPUT_PATH, index=False)
    result["test_results"].to_csv(TEST_OUTPUT_PATH, index=False)

    print("\nSaved:")
    print(f"- {VAL_OUTPUT_PATH}")
    print(f"- {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()