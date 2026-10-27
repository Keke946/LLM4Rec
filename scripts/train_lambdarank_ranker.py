from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.ranking.lambdarank import (
    build_feature_schema,
    grouped_ltr_frame,
    lightgbm_available,
    make_lambdarank_features,
    make_relevance_labels,
    save_lambdarank_model,
    validation_ranking_metrics,
)


from llm4rec.validation import validate_training_splits

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM LambdaRank candidate reranker.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--query_col", default="user_id")
    parser.add_argument("--label_mode", choices=["binary", "graded"], default="graded")
    parser.add_argument("--n_estimators", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--eval_at", default="5,10,20,50,100")
    parser.add_argument(
        "--include_injected_positives",
        action="store_true",
        help="Include audited injected positives. Default excludes them from model training.",
    )
    args = parser.parse_args()
    validate_training_splits(args.train_split, args.val_split)

    if not lightgbm_available():
        raise RuntimeError(
            "LightGBM is required for LambdaRank training. Install `lightgbm` or run non-LTR tests only."
        )
    from lightgbm import LGBMRanker

    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    output_dir = Path(args.output_dir or f"models/{args.dataset}/lambdarank")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_at = [int(value.strip()) for value in args.eval_at.split(",") if value.strip()]

    train = _load_samples(features_dir / f"{args.train_split}_rank_samples.parquet", args.include_injected_positives)
    val = _load_samples(features_dir / f"{args.val_split}_rank_samples.parquet", args.include_injected_positives)
    train = _sample_group_rows(train, args.query_col, args.max_train_rows, args.random_state)
    val = _sample_group_rows(val, args.query_col, args.max_val_rows, args.random_state + 1)
    if train.empty or val.empty:
        raise RuntimeError("LambdaRank requires non-empty train and validation candidate samples.")

    schema = build_feature_schema(pd.concat([train, val], ignore_index=True))
    train_labels_raw = make_relevance_labels(train, args.label_mode)
    val_labels_raw = make_relevance_labels(val, args.label_mode)
    train_grouped, y_train, train_groups = grouped_ltr_frame(
        train,
        train_labels_raw,
        query_col=args.query_col,
        min_group_size=2,
        require_positive=True,
    )
    val_grouped, y_val, val_groups = grouped_ltr_frame(
        val,
        val_labels_raw,
        query_col=args.query_col,
        min_group_size=2,
        require_positive=True,
    )
    if train_grouped.empty or val_grouped.empty:
        raise RuntimeError("LambdaRank grouping removed all rows; check candidate positives and query column.")

    x_train = make_lambdarank_features(train_grouped, schema)
    x_val = make_lambdarank_features(val_grouped, schema)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        verbose=-1,
    )
    print(
        f"[lambdarank] fitting rows={len(train_grouped)} groups={len(train_groups)} "
        f"val_rows={len(val_grouped)} val_groups={len(val_groups)}",
        file=sys.stderr,
        flush=True,
    )
    model.fit(
        x_train,
        y_train,
        group=train_groups,
        eval_set=[(x_val, y_val)],
        eval_group=[val_groups],
        eval_at=eval_at,
    )
    scores = model.predict(x_val)
    metrics = validation_ranking_metrics(val_grouped, scores, eval_at, query_col=args.query_col)
    metrics.update(
        {
            "train_rows": int(len(train_grouped)),
            "val_rows": int(len(val_grouped)),
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "train_positive_rate": float((y_train > 0).mean()) if len(y_train) else 0.0,
            "val_positive_rate": float((y_val > 0).mean()) if len(y_val) else 0.0,
            "label_mode": args.label_mode,
        }
    )
    config: dict[str, Any] = {
        "model": "lambdarank",
        "dataset": args.dataset,
        "features_dir": str(features_dir),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "query_col": args.query_col,
        "label_mode": args.label_mode,
        "eval_at": eval_at,
        "exclude_injected_positives": not args.include_injected_positives,
        "feature_columns": schema["feature_columns"],
        "source_names": schema["source_names"],
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "random_state": args.random_state,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
    }
    save_lambdarank_model(model, output_dir, schema)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics_val.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config": config, "metrics": metrics}, ensure_ascii=False, indent=2))


def _load_samples(path: Path, include_injected_positives: bool) -> pd.DataFrame:
    import pyarrow.parquet as pq
    excluded = {"history_item_indices", "retrieval_source_indices", "timestamp", "split"}
    columns = [c for c in pq.read_schema(path).names if c not in excluded]
    frame = pd.read_parquet(path, columns=columns).reset_index(drop=True)
    if not include_injected_positives and "is_injected_positive" in frame.columns:
        labels = pd.to_numeric(frame["label"], errors="coerce").fillna(0)
        injected = pd.to_numeric(frame["is_injected_positive"], errors="coerce").fillna(0)
        frame = frame[~((labels > 0) & (injected > 0))].reset_index(drop=True)
    return frame


def _sample_group_rows(frame: pd.DataFrame, query_col: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or max_rows >= len(frame) or query_col not in frame.columns:
        return frame
    rng = pd.Series(sorted(frame[query_col].astype(str).unique())).sample(frac=1.0, random_state=seed)
    selected_users: list[str] = []
    rows = 0
    counts = frame.groupby(frame[query_col].astype(str)).size().to_dict()
    for user_id in rng.tolist():
        count = int(counts.get(user_id, 0))
        if selected_users and rows + count > max_rows:
            break
        selected_users.append(user_id)
        rows += count
        if rows >= max_rows:
            break
    return frame[frame[query_col].astype(str).isin(selected_users)].reset_index(drop=True)


if __name__ == "__main__":
    main()
