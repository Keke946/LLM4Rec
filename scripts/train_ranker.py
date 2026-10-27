from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

from llm4rec.features import FeatureEncoders, RankingSampleDataset
from llm4rec.ranking import RANKER_REGISTRY
from llm4rec.ranking.base import RankerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ranking model.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--model", required=True, choices=sorted(RANKER_REGISTRY), help="Ranker name.")
    parser.add_argument("--features_dir", default=None, help="Feature directory.")
    parser.add_argument("--output_dir", default=None, help="Model output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--max_history_len", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_rows", type=int, default=None, help="Optional reproducible cap for fast ablations.")
    parser.add_argument("--max_val_rows", type=int, default=None, help="Optional reproducible cap for fast ablations.")
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.sample_seed)
    np.random.seed(args.sample_seed)

    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    output_dir = Path(args.output_dir or f"models/{args.dataset}/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = FeatureEncoders.load(features_dir / "encoders.pkl")
    config = RankerConfig(
        num_users=len(encoders.user),
        num_items=len(encoders.item),
        num_authors=len(encoders.author),
        num_categories=len(encoders.category),
        num_tags=len(encoders.tag),
        num_retrieval_sources=len(encoders.retrieval_source),
        num_duration_buckets=len(encoders.duration_boundaries) + 2,
        dense_feature_dim=6,
        embedding_dim=args.embedding_dim,
        max_history_len=args.max_history_len,
    )
    model = RANKER_REGISTRY[args.model](config).to(args.device)

    train_ds = _maybe_subset(
        RankingSampleDataset(features_dir / "train_rank_samples.parquet", args.max_history_len),
        args.max_train_rows,
        args.sample_seed,
    )
    val_ds = _maybe_subset(
        RankingSampleDataset(features_dir / "val_rank_samples.parquet", args.max_history_len),
        args.max_val_rows,
        args.sample_seed + 1,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.sample_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_metrics: dict[str, float | int | None] = {}
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, args.device)
            logits = model(batch)
            if torch.isnan(logits).any():
                raise RuntimeError("NaN detected in ranker logits.")
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val_loader, args.device)
        train_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"val_auc={_fmt(val_metrics['val_auc'])} "
            f"val_logloss={_fmt(val_metrics['val_logloss'])}"
        )
        score = val_metrics["val_auc"] if val_metrics["val_auc"] is not None else -val_metrics["val_logloss"]
        if score > best_auc:
            best_auc = float(score)
            best_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            torch.save(model.state_dict(), output_dir / "model.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    config_dict = asdict(config)
    config_dict.update(
        {
            "model": args.model,
            "dataset": args.dataset,
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "sample_seed": args.sample_seed,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best {args.model} model to {output_dir}")


def _maybe_subset(dataset: Dataset, max_rows: int | None, seed: int) -> Dataset:
    if max_rows is None or max_rows <= 0 or max_rows >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_rows, replace=False)
    return Subset(dataset, indices.tolist())


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float | None]:
    model.eval()
    labels: list[float] = []
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            labels.extend(batch["label"].detach().cpu().numpy().tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    if not labels:
        return {"val_auc": None, "val_logloss": None}
    unique = set(int(label) for label in labels)
    auc = float(roc_auc_score(labels, probs)) if len(unique) > 1 else None
    ll = float(log_loss(labels, probs, labels=[0, 1]))
    return {"val_auc": auc, "val_logloss": ll}


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
