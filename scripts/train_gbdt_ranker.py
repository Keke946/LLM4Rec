from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from llm4rec.evaluation.metrics import ranking_metrics
from llm4rec.ranking.gbdt_features import feature_columns, infer_source_names, make_gbdt_features


from llm4rec.validation import validate_training_splits

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight GBDT candidate reranker.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.06)
    parser.add_argument("--max_leaf_nodes", type=int, default=31)
    parser.add_argument("--l2_regularization", type=float, default=0.01)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--include_injected_positives",
        action="store_true",
        help="Include audited injected positives. Default excludes them because they are not live candidates.",
    )
    args = parser.parse_args()
    validate_training_splits(args.train_split, args.val_split)

    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    output_dir = Path(args.output_dir or f"models/{args.dataset}/gbdt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[gbdt] loading samples train_split={args.train_split} val_split={args.val_split}",
        file=sys.stderr,
        flush=True,
    )
    train = _load_samples(features_dir / f"{args.train_split}_rank_samples.parquet", args.include_injected_positives)
    val = _load_samples(features_dir / f"{args.val_split}_rank_samples.parquet", args.include_injected_positives)
    train = _sample_rows(train, args.max_train_rows, args.sample_seed)
    val = _sample_rows(val, args.max_val_rows, args.sample_seed + 1)
    if train.empty or val.empty:
        raise RuntimeError("GBDT ranker requires non-empty train and validation candidate samples.")

    print(
        f"[gbdt] building features train_rows={len(train)} val_rows={len(val)}",
        file=sys.stderr,
        flush=True,
    )
    source_names = infer_source_names(pd.concat([train, val], ignore_index=True))
    columns = feature_columns(source_names)
    x_train = make_gbdt_features(train, source_names)
    y_train = train["label"].astype(int).to_numpy()
    x_val = make_gbdt_features(val, source_names)
    y_val = val["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) != 2:
        raise ValueError("GBDT requires both positive and negative live train candidates; build temporal train samples.")
    sample_weight = _balanced_sample_weight(y_train)

    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.sample_seed,
    )
    print("[gbdt] fitting HistGradientBoostingClassifier", file=sys.stderr, flush=True)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    print("[gbdt] evaluating validation split", file=sys.stderr, flush=True)
    val_prob = model.predict_proba(x_val)[:, 1]
    metrics = _classification_metrics(y_val, val_prob)
    metrics.update(_ranking_eval(val, val_prob))
    metrics.update(
        {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "train_positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "val_positive_rate": float(y_val.mean()) if len(y_val) else 0.0,
        }
    )

    with (output_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    config = {
        "model": "gbdt",
        "dataset": args.dataset,
        "feature_columns": columns,
        "source_names": source_names,
        "exclude_injected_positives": not args.include_injected_positives,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "sample_seed": args.sample_seed,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "train_split": args.train_split,
        "val_split": args.val_split,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config": config, "metrics": metrics}, ensure_ascii=False, indent=2))


def _load_samples(path: Path, include_injected_positives: bool) -> pd.DataFrame:
    import pyarrow.parquet as pq
    excluded = {"history_item_indices", "retrieval_source_indices", "timestamp", "split"}
    columns = [c for c in pq.read_schema(path).names if c not in excluded]
    frame = pd.read_parquet(path, columns=columns).reset_index(drop=True)
    if not include_injected_positives and "is_injected_positive" in frame.columns:
        frame = frame[~((frame["label"] == 1) & (frame["is_injected_positive"] == 1))].reset_index(drop=True)
    return frame


def _sample_rows(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or max_rows >= len(frame):
        return frame
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    weights = np.ones(len(labels), dtype=np.float32)
    if positives > 0 and negatives > 0:
        weights[labels == 1] = negatives / positives
    return weights


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    unique = set(int(value) for value in labels)
    auc = float(roc_auc_score(labels, probabilities)) if len(unique) > 1 else None
    loss = float(log_loss(labels, probabilities, labels=[0, 1]))
    return {"val_auc": auc, "val_logloss": loss}


def _ranking_eval(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    scored = frame[["user_id", "item_id", "label"]].copy()
    scored["score"] = probabilities
    rows: list[dict[str, float]] = []
    for user_id, group in scored.groupby("user_id"):
        relevant = set(group[group["label"] == 1]["item_id"].astype(str))
        if not relevant:
            continue
        ranked = group.sort_values("score", ascending=False)["item_id"].astype(str).tolist()
        rows.append(ranking_metrics(ranked, relevant, [10, 50, 100]))
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {f"val_{key}": float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


if __name__ == "__main__":
    main()
