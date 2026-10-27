from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.retrieval import RETRIEVER_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/build a retrieval model.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--model", required=True, choices=sorted(RETRIEVER_REGISTRY), help="Retriever name.")
    parser.add_argument("--processed_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--output_dir", default=None, help="Retriever output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for trainable retrievers.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for trainable retrievers.")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dim for trainable retrievers.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for trainable retrievers.")
    parser.add_argument("--max_history_len", type=int, default=20, help="History length for TwoTower/SASRec.")
    parser.add_argument("--num_layers", type=int, default=2, help="Propagation layers for LightGCN.")
    parser.add_argument("--trend_half_life_days", type=float, default=2.0, help="Trend retriever half-life in days.")
    parser.add_argument("--trend_category_weight", type=float, default=0.4, help="Trend retriever category affinity weight.")
    parser.add_argument("--trend_tag_weight", type=float, default=0.2, help="Trend retriever tag affinity weight.")
    parser.add_argument("--trend_popularity_floor", type=float, default=0.02, help="Trend retriever popularity floor.")
    parser.add_argument("--trend_candidate_cache_k", type=int, default=2000, help="Items personalized per user by Trend retriever.")
    parser.add_argument("--exposure_half_life_days", type=float, default=1.0, help="Exposure retriever half-life in days.")
    parser.add_argument("--exposure_watch_ratio_power", type=float, default=1.0, help="Exposure retriever watch-ratio exponent.")
    parser.add_argument("--exposure_positive_boost", type=float, default=0.5, help="Exposure retriever positive-label score boost.")
    parser.add_argument("--exposure_score_floor", type=float, default=0.001, help="Exposure retriever minimum item score.")
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu or cuda.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    output_dir = Path(args.output_dir or f"models/{args.dataset}/retrievers")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "items.parquet")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    train = interactions[interactions["split"] == "train"].reset_index(drop=True)
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)

    retriever_cls = RETRIEVER_REGISTRY[args.model]
    kwargs = {}
    if args.model in {"bpr", "two_tower", "lightgcn", "sasrec"}:
        kwargs = {
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "device": args.device,
        }
    if args.model == "two_tower":
        kwargs["max_history_len"] = args.max_history_len
    if args.model == "sasrec":
        kwargs["max_seq_len"] = args.max_history_len
    if args.model == "lightgcn":
        kwargs["num_layers"] = args.num_layers
    if args.model == "trend":
        kwargs = {
            "half_life_days": args.trend_half_life_days,
            "category_weight": args.trend_category_weight,
            "tag_weight": args.trend_tag_weight,
            "popularity_floor": args.trend_popularity_floor,
            "candidate_cache_k": args.trend_candidate_cache_k,
        }
    if args.model == "exposure":
        kwargs = {
            "half_life_days": args.exposure_half_life_days,
            "watch_ratio_power": args.exposure_watch_ratio_power,
            "positive_boost": args.exposure_positive_boost,
            "score_floor": args.exposure_score_floor,
        }
    retriever = retriever_cls(**kwargs).fit(train, items)
    path = output_dir / args.model
    retriever.save(path)
    metrics = _validation_recall(retriever, val)
    if metrics:
        (path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved {args.model} retriever to {path}")


def _validation_recall(retriever, val: pd.DataFrame, cutoffs: tuple[int, ...] = (50, 100, 200)) -> dict[str, float]:
    positives = val[val["label"] == 1]
    if positives.empty:
        return {}
    relevant_by_user = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in positives.groupby("user_id")
    }
    totals = {cutoff: 0.0 for cutoff in cutoffs}
    users = 0
    max_k = max(cutoffs)
    for user_id, relevant in relevant_by_user.items():
        relevant = relevant - retriever.train_positive_by_user.get(user_id, set())
        if not relevant:
            continue
        recs = [candidate.item_id for candidate in retriever.retrieve(user_id, top_k=max_k)]
        users += 1
        for cutoff in cutoffs:
            totals[cutoff] += len(set(recs[:cutoff]) & relevant) / len(relevant)
    if users == 0:
        return {}
    metrics = {f"val_recall@{cutoff}": totals[cutoff] / users for cutoff in cutoffs}
    metrics["val_users"] = users
    return metrics


if __name__ == "__main__":
    main()
