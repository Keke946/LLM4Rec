from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.features import FeatureEncoders
from llm4rec.features.build_features import _encode_item_lookup, _prepare_item_features
from llm4rec.retrieval import RETRIEVER_REGISTRY

from scripts.build_ranker_candidates import (
    OUTPUT_COLUMNS,
    _history_indices,
    _history_profiles,
    _parse_source_weights,
    _positive_items_by_user,
    _retriever_kwargs,
    _rows_for_user,
    _split_timestamps,
)

from llm4rec.features.candidate_writer import CandidateWriter


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ranker train samples from an internal temporal holdout inside the train split."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--retrievers", default="trend")
    parser.add_argument("--candidate_pool_k", type=int, default=1000)
    parser.add_argument("--retriever_top_k", type=int, default=None)
    parser.add_argument("--holdout_quantile", type=float, default=0.8)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--source_weights", default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--trend_half_life_days", type=float, default=2.0)
    parser.add_argument("--trend_category_weight", type=float, default=0.4)
    parser.add_argument("--trend_tag_weight", type=float, default=0.2)
    parser.add_argument("--trend_popularity_floor", type=float, default=0.02)
    parser.add_argument("--trend_candidate_cache_k", type=int, default=2000)
    parser.add_argument("--exposure_half_life_days", type=float, default=1.0)
    parser.add_argument("--exposure_watch_ratio_power", type=float, default=1.0)
    parser.add_argument("--exposure_positive_boost", type=float, default=0.5)
    parser.add_argument("--exposure_score_floor", type=float, default=0.001)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    output_dir = Path(args.output_dir or features_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "items.parquet")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    train = interactions[interactions["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("No train split found.")

    cutoff = float(pd.to_numeric(train["timestamp"], errors="coerce").quantile(args.holdout_quantile))
    prefix = train[pd.to_numeric(train["timestamp"], errors="coerce") <= cutoff].copy()
    target = train[pd.to_numeric(train["timestamp"], errors="coerce") > cutoff].copy()
    prefix["split"] = "prefix"
    target["split"] = "temporal_train"
    if prefix.empty or target.empty:
        raise RuntimeError("Temporal train split produced empty prefix or target.")

    encoders = FeatureEncoders.load(features_dir / "encoders.pkl")
    output_encoders_path = output_dir / "encoders.pkl"
    if output_encoders_path.resolve() != (features_dir / "encoders.pkl").resolve():
        encoders.save(output_encoders_path)
    item_lookup = _encode_item_lookup(_prepare_item_features(items), encoders)

    retriever_names = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    unknown = sorted(set(retriever_names) - set(RETRIEVER_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown retrievers: {unknown}")
    retrievers = {
        name: RETRIEVER_REGISTRY[name](**_retriever_kwargs(name, args)).fit(prefix, items)
        for name in retriever_names
    }

    relevant_by_user = _positive_items_by_user(target, ["temporal_train"])
    seen_by_user = _positive_items_by_user(prefix, ["prefix"])
    item_popularity = {
        str(item_id): int(count)
        for item_id, count in prefix[prefix["label"] == 1].groupby("item_id").size().items()
    }
    user_activity = {
        str(user_id): int(count)
        for user_id, count in prefix[prefix["label"] == 1].groupby("user_id").size().items()
    }
    histories = _history_indices(prefix, ["prefix"], encoders, args.max_history_len)
    history_profiles = _history_profiles(prefix, ["prefix"], item_lookup, args.max_history_len)
    timestamps = _split_timestamps(target).get("temporal_train", {})
    retriever_top_k = args.retriever_top_k or args.candidate_pool_k
    source_weights = _parse_source_weights(args.source_weights)

    writer = CandidateWriter(output_dir / "train_rank_samples.parquet", OUTPUT_COLUMNS)
    user_ids = sorted(relevant_by_user)
    if args.max_users is not None:
        user_ids = user_ids[: args.max_users]
    for user_id in user_ids:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue
        writer.append(
            _rows_for_user(
                user_id=user_id,
                split="train",
                relevant=relevant,
                seen=seen_by_user.get(user_id, set()),
                retrievers=retrievers,
                item_lookup=item_lookup,
                encoders=encoders,
                candidate_pool_k=args.candidate_pool_k,
                retriever_top_k=retriever_top_k,
                source_weights=source_weights,
                history=histories.get(user_id, []),
                history_profile=history_profiles.get(user_id, {}),
                item_popularity=item_popularity,
                user_activity=user_activity,
                timestamp=timestamps.get(user_id, 0),
            )
        )

    split_stats = writer.finish()
    meta_path = output_dir / "ranker_candidate_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(
        {
            "dataset": args.dataset,
            "temporal_train": {
                "holdout_quantile": args.holdout_quantile,
                "cutoff_timestamp": cutoff,
                "prefix_rows": int(len(prefix)),
                "target_rows": int(len(target)),
                "retrievers": retriever_names,
                "candidate_pool_k": args.candidate_pool_k,
                "retriever_top_k": retriever_top_k,
                "source_weights": source_weights,
            },
        }
    )
    splits = meta.setdefault("splits", {})
    splits["train"] = split_stats
    _refresh_overall(meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s rows to %s", split_stats["rows"], output_dir / "train_rank_samples.parquet")
    print(f"Built temporal train ranker samples at {output_dir}")


def _split_stats(frame: pd.DataFrame, users: int) -> dict[str, Any]:
    positives = int(frame["label"].sum()) if not frame.empty else 0
    injected = int(frame["is_injected_positive"].sum()) if not frame.empty else 0
    retrieved = int(frame["is_retrieved_positive"].sum()) if not frame.empty else 0
    return {
        "rows": int(len(frame)),
        "positives": positives,
        "retrieved_positives": retrieved,
        "injected_positives": injected,
        "positive_injection_rate": float(injected / positives) if positives else 0.0,
        "candidate_positive_recall": float(retrieved / positives) if positives else 0.0,
        "negatives": int((frame["label"] == 0).sum()) if not frame.empty else 0,
        "users": int(users),
    }


def _refresh_overall(meta: dict[str, Any]) -> None:
    splits = meta.get("splits", {})
    positives = sum(int(stats.get("positives", 0)) for stats in splits.values())
    retrieved = sum(int(stats.get("retrieved_positives", 0)) for stats in splits.values())
    injected = sum(int(stats.get("injected_positives", 0)) for stats in splits.values())
    meta["overall"] = {
        "positives": positives,
        "retrieved_positives": retrieved,
        "injected_positives": injected,
        "positive_injection_rate": float(injected / positives) if positives else 0.0,
        "candidate_positive_recall": float(retrieved / positives) if positives else 0.0,
    }


if __name__ == "__main__":
    main()
