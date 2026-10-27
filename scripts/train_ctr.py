from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from llm4rec.config import DEFAULT_PATHS, ensure_project_dirs
from llm4rec.models import CTRMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def train_ctr_model(
    samples_csv: str,
    model_output: str,
    metrics_output: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float | int | None]:
    ensure_project_dirs()
    set_seed(seed)

    train_df = pd.read_csv(samples_csv)
    if "label" not in train_df.columns:
        raise ValueError("CTR samples must include a `label` column.")

    X = train_df.drop(columns=["label"]).astype("float32").values
    y = train_df["label"].astype("float32").values
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = CTRMLP(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        last_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).detach().numpy().reshape(-1)
        val_auc = safe_auc(y_val, val_preds)
        val_logloss = float(log_loss(y_val, val_preds, labels=[0, 1]))
        auc_text = f"{val_auc:.4f}" if val_auc is not None else "n/a"
        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"train_loss={last_loss:.4f} val_auc={auc_text} val_logloss={val_logloss:.4f}"
        )

    torch.save(model.state_dict(), model_output)
    metrics = {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "input_dim": int(X.shape[1]),
        "epochs": int(epochs),
        "train_loss": last_loss,
        "val_auc": val_auc,
        "val_logloss": val_logloss,
    }
    with open(metrics_output, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CTR MLP ranker.")
    parser.add_argument("--samples", default=str(DEFAULT_PATHS.ctr_samples_csv), help="CTR samples CSV.")
    parser.add_argument("--model-output", default=str(DEFAULT_PATHS.ctr_model), help="Model output path.")
    parser.add_argument("--metrics-output", default=str(DEFAULT_PATHS.ctr_metrics), help="Metrics JSON path.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    metrics = train_ctr_model(
        samples_csv=args.samples,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"Saved model to {args.model_output}")
    print(f"Saved metrics to {args.metrics_output}: {metrics}")


if __name__ == "__main__":
    main()
