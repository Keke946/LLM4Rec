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
from llm4rec.retrieval import RETRIEVER_REGISTRY, reciprocal_rank_fusion
from llm4rec.retrieval.base import BaseRetriever, Candidate


logger = logging.getLogger(__name__)

DEFAULT_RETRIEVERS = [
    "popular",
    "itemcf",
    "bpr",
    "lightgcn",
    "two_tower",
    "sasrec",
    "content_faiss",
    "metadata",
    "trend",
    "exposure",
]

OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "user_idx",
    "item_idx",
    "label",
    "is_injected_positive",
    "is_retrieved_positive",
    "category_indices",
    "tag_indices",
    "author_idx",
    "duration_bucket",
    "history_item_indices",
    "history_len",
    "retrieval_score",
    "retrieval_source",
    "retrieval_sources",
    "retrieval_source_indices",
    "source",
    "source_rank",
    "source_count",
    "rank",
    "candidate_pool_size",
    "item_popularity",
    "user_activity_level",
    "category_overlap",
    "tag_overlap",
    "category_jaccard",
    "tag_jaccard",
    "history_category_count",
    "history_tag_count",
    "timestamp",
    "split",
]


from llm4rec.features.candidate_writer import CandidateWriter

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-level ranker samples from real retriever outputs."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--processed_dir", default=None, help="Processed parquet directory.")
    parser.add_argument("--features_dir", default=None, help="Feature directory containing encoders.pkl.")
    parser.add_argument("--retriever_dir", default=None, help="Retriever artifact directory.")
    parser.add_argument("--output_dir", default=None, help="Output directory for rank samples.")
    parser.add_argument(
        "--retrievers",
        default=",".join(DEFAULT_RETRIEVERS),
        help="Comma-separated retrievers. If fusion is included it is treated as the merged output.",
    )
    parser.add_argument("--candidate_pool_k", type=int, default=500)
    parser.add_argument("--retriever_top_k", type=int, default=None)
    parser.add_argument(
        "--source_weights",
        default=None,
        help="Optional comma-separated RRF source weights, e.g. sasrec=1.2,metadata=0.4.",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--max_users_per_split", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for trainable retrievers if refit is needed.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_layers", type=int, default=2, help="LightGCN propagation layers.")
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
    parser.add_argument(
        "--force_retrain_retrievers",
        action="store_true",
        help="Ignore existing retriever artifacts and fit fresh retrievers.",
    )
    parser.add_argument(
        "--no_save_retrievers",
        action="store_true",
        help="Do not persist fitted retrievers to retriever_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    retriever_dir = Path(args.retriever_dir or f"models/{args.dataset}/retrievers")
    output_dir = Path(args.output_dir or features_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "items.parquet")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    train = interactions[interactions["split"] == "train"].reset_index(drop=True)
    encoders = FeatureEncoders.load(features_dir / "encoders.pkl")
    output_encoders_path = output_dir / "encoders.pkl"
    if output_encoders_path.resolve() != (features_dir / "encoders.pkl").resolve():
        encoders.save(output_encoders_path)
    item_lookup = _encode_item_lookup(_prepare_item_features(items), encoders)
    retriever_names = _normalize_retrievers(_split(args.retrievers))
    retriever_top_k = args.retriever_top_k or args.candidate_pool_k
    source_weights = _parse_source_weights(args.source_weights)

    retrievers = {
        name: _load_or_fit_retriever(
            name=name,
            train=train,
            items=items,
            retriever_dir=retriever_dir,
            args=args,
        )
        for name in retriever_names
    }

    split_names = _split(args.splits)
    positives_by_split = {
        split: _positive_items_by_user(interactions, [split])
        for split in split_names
    }
    train_positive_by_user = _positive_items_by_user(interactions, ["train"])
    val_positive_by_user = _positive_items_by_user(interactions, ["val"])
    item_popularity = {
        str(item_id): int(count)
        for item_id, count in train[train["label"] == 1].groupby("item_id").size().items()
    }
    user_activity = {
        str(user_id): int(count)
        for user_id, count in train[train["label"] == 1].groupby("user_id").size().items()
    }
    histories = _history_by_split(interactions, encoders, args.max_history_len)
    history_profiles = _history_profiles_by_split(interactions, item_lookup, args.max_history_len)
    split_timestamps = _split_timestamps(interactions)

    meta: dict[str, Any] = {
        "dataset": args.dataset,
        "retrievers": retriever_names,
        "candidate_pool_k": args.candidate_pool_k,
        "retriever_top_k": retriever_top_k,
        "source_weights": source_weights,
        "splits": {},
        "note": "Candidate-level ranker samples generated from retriever outputs; missing positives are injected as audited labels, not source tokens.",
    }

    for split in split_names:
        relevant_by_user = positives_by_split.get(split, {})
        writer = CandidateWriter(output_dir / f"{split}_rank_samples.parquet", OUTPUT_COLUMNS)
        user_ids = sorted(relevant_by_user)
        if args.max_users_per_split is not None:
            user_ids = user_ids[: args.max_users_per_split]
        for user_id in user_ids:
            relevant = relevant_by_user.get(user_id, set())
            if not relevant:
                continue
            seen = _seen_for_split(split, train_positive_by_user, val_positive_by_user)
            seen_for_user = seen.get(user_id, set())
            writer.append(
                _rows_for_user(
                    user_id=user_id,
                    split=split,
                    relevant=relevant,
                    seen=seen_for_user,
                    retrievers=retrievers,
                    item_lookup=item_lookup,
                    encoders=encoders,
                    candidate_pool_k=args.candidate_pool_k,
                    retriever_top_k=retriever_top_k,
                    source_weights=source_weights,
                    history=histories.get(split, {}).get(user_id, []),
                    history_profile=history_profiles.get(split, {}).get(user_id, {}),
                    item_popularity=item_popularity,
                    user_activity=user_activity,
                    timestamp=split_timestamps.get(split, {}).get(user_id, 0),
                )
            )
        meta["splits"][split] = writer.finish()
        logger.info("Wrote %s rows for split=%s", meta["splits"][split]["rows"], split)

    totals = {
        "positives": sum(int(stats.get("positives", 0)) for stats in meta["splits"].values()),
        "retrieved_positives": sum(
            int(stats.get("retrieved_positives", 0)) for stats in meta["splits"].values()
        ),
        "injected_positives": sum(
            int(stats.get("injected_positives", 0)) for stats in meta["splits"].values()
        ),
    }
    positives = totals["positives"]
    totals["positive_injection_rate"] = float(totals["injected_positives"] / positives) if positives else 0.0
    totals["candidate_positive_recall"] = float(totals["retrieved_positives"] / positives) if positives else 0.0
    meta["overall"] = totals

    (output_dir / "ranker_candidate_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Built candidate-level ranker samples at {output_dir}")


def _load_or_fit_retriever(
    name: str,
    train: pd.DataFrame,
    items: pd.DataFrame,
    retriever_dir: Path,
    args: argparse.Namespace,
) -> BaseRetriever:
    if not args.force_retrain_retrievers:
        path = retriever_dir / name
        if path.exists():
            try:
                retriever = BaseRetriever.load(path)
                retriever.items = items.copy()
                retriever.train_interactions = train.copy()
                logger.info("Loaded retriever %s from %s", name, path)
                return retriever
            except Exception as exc:
                logger.warning("Failed to load retriever %s from %s: %s. Refitting.", name, path, exc)

    retriever_cls = RETRIEVER_REGISTRY[name]
    retriever = retriever_cls(**_retriever_kwargs(name, args)).fit(train, items)
    if not args.no_save_retrievers:
        retriever_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(retriever_dir / name)
        logger.info("Saved retriever %s to %s", name, retriever_dir / name)
    return retriever


def _retriever_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "trend":
        return {
            "half_life_days": args.trend_half_life_days,
            "category_weight": args.trend_category_weight,
            "tag_weight": args.trend_tag_weight,
            "popularity_floor": args.trend_popularity_floor,
            "candidate_cache_k": args.trend_candidate_cache_k,
        }
    if name == "exposure":
        return {
            "half_life_days": args.exposure_half_life_days,
            "watch_ratio_power": args.exposure_watch_ratio_power,
            "positive_boost": args.exposure_positive_boost,
            "score_floor": args.exposure_score_floor,
        }
    if name not in {"bpr", "lightgcn", "two_tower", "sasrec"}:
        return {}
    kwargs: dict[str, Any] = {
        "epochs": args.epochs,
        "embedding_dim": args.embedding_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    if name == "two_tower":
        kwargs["max_history_len"] = args.max_history_len
    if name == "sasrec":
        kwargs["max_seq_len"] = args.max_history_len
    if name == "lightgcn":
        kwargs["num_layers"] = args.num_layers
    return kwargs


def _rows_for_user(
    user_id: str,
    split: str,
    relevant: set[str],
    seen: set[str],
    retrievers: dict[str, BaseRetriever],
    item_lookup: dict[str, dict[str, Any]],
    encoders: FeatureEncoders,
    candidate_pool_k: int,
    retriever_top_k: int,
    source_weights: dict[str, float] | None,
    history: list[int],
    history_profile: dict[str, set[int]],
    item_popularity: dict[Any, int],
    user_activity: dict[Any, int],
    timestamp: int,
) -> list[dict[str, Any]]:
    candidate_lists: list[list[Candidate]] = []
    expanded_retriever_top_k = retriever_top_k + len(seen)
    for retriever in retrievers.values():
        try:
            candidates = retriever.retrieve(user_id, top_k=expanded_retriever_top_k)
        except Exception as exc:
            logger.warning("Retriever %s failed for user=%s: %s", retriever.name, user_id, exc)
            candidates = []
        candidate_lists.append(_filter_candidates(candidates, seen))

    merged = reciprocal_rank_fusion(candidate_lists, top_k=candidate_pool_k, weights=source_weights)
    merged = _filter_candidates(merged, seen)
    by_item = {candidate.item_id: candidate for candidate in merged}
    retrieved_item_ids = set(by_item)
    # This is a model feature: it must not depend on the target labels injected below.
    candidate_pool_size = len(by_item)
    next_rank = len(by_item) + 1
    for item_id in sorted(relevant):
        if item_id in seen:
            continue
        if item_id not in by_item:
            by_item[item_id] = Candidate(
                item_id=item_id,
                score=0.0,
                source="",
                rank=next_rank,
                retrieval_score=0.0,
            )
            next_rank += 1

    ordered = sorted(by_item.values(), key=lambda candidate: (candidate.rank, candidate.item_id))
    rows: list[dict[str, Any]] = []
    for candidate in ordered:
        item_info = item_lookup.get(candidate.item_id)
        if item_info is None:
            continue
        sources = _sources(candidate)
        retrieval_source_indices = encoders.retrieval_source.transform_list(sources, max_len=8)
        retrieval_score = float(
            candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score
        )
        is_positive = candidate.item_id in relevant
        is_injected_positive = is_positive and candidate.item_id not in retrieved_item_ids
        is_retrieved_positive = is_positive and candidate.item_id in retrieved_item_ids
        feature_source_rank = 0.0 if is_injected_positive else float(candidate.rank)
        cross_features = _content_cross_features(history_profile, item_info)
        rows.append(
            {
                "user_id": user_id,
                "item_id": candidate.item_id,
                "user_idx": encoders.user.transform_one(user_id),
                "item_idx": item_info["item_idx"],
                "label": int(is_positive),
                "is_injected_positive": int(is_injected_positive),
                "is_retrieved_positive": int(is_retrieved_positive),
                "category_indices": item_info["category_indices"],
                "tag_indices": item_info["tag_indices"],
                "author_idx": item_info["author_idx"],
                "duration_bucket": item_info["duration_bucket"],
                "history_item_indices": history,
                "history_len": len(history),
                "retrieval_score": retrieval_score,
                "retrieval_source": "+".join(sources),
                "retrieval_sources": sources,
                "retrieval_source_indices": retrieval_source_indices,
                "source": "+".join(sources),
                "source_rank": feature_source_rank,
                "source_count": float(len(sources)),
                "rank": int(candidate.rank),
                "candidate_pool_size": candidate_pool_size,
                "item_popularity": float(item_popularity.get(candidate.item_id, 0)),
                "user_activity_level": float(user_activity.get(user_id, 0)),
                **cross_features,
                "timestamp": int(timestamp),
                "split": split,
            }
        )
    return rows


def _normalize_retrievers(names: list[str]) -> list[str]:
    requested = [name for name in names if name != "fusion"]
    if not requested:
        requested = DEFAULT_RETRIEVERS
    unknown = sorted(set(requested) - set(RETRIEVER_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown retrievers: {unknown}")
    return requested


def _positive_items_by_user(interactions: pd.DataFrame, splits: list[str]) -> dict[str, set[str]]:
    positives = interactions[
        (interactions["split"].isin(splits)) & (interactions["label"] == 1)
    ]
    return {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in positives.groupby("user_id")
    }


def _seen_for_split(
    split: str,
    train_positive_by_user: dict[str, set[str]],
    val_positive_by_user: dict[str, set[str]],
) -> dict[str, set[str]]:
    if split == "train":
        return {}
    if split == "val":
        return train_positive_by_user
    seen = {user_id: set(items) for user_id, items in train_positive_by_user.items()}
    for user_id, items in val_positive_by_user.items():
        seen.setdefault(user_id, set()).update(items)
    return seen


def _history_by_split(
    interactions: pd.DataFrame,
    encoders: FeatureEncoders,
    max_history_len: int,
) -> dict[str, dict[str, list[int]]]:
    return {
        "train": _history_indices(interactions, ["train"], encoders, max_history_len),
        "val": _history_indices(interactions, ["train"], encoders, max_history_len),
        "test": _history_indices(interactions, ["train", "val"], encoders, max_history_len),
    }


def _history_profiles_by_split(
    interactions: pd.DataFrame,
    item_lookup: dict[str, dict[str, Any]],
    max_history_len: int,
) -> dict[str, dict[str, dict[str, set[int]]]]:
    return {
        "train": _history_profiles(interactions, ["train"], item_lookup, max_history_len),
        "val": _history_profiles(interactions, ["train"], item_lookup, max_history_len),
        "test": _history_profiles(interactions, ["train", "val"], item_lookup, max_history_len),
    }


def _history_profiles(
    interactions: pd.DataFrame,
    splits: list[str],
    item_lookup: dict[str, dict[str, Any]],
    max_history_len: int,
) -> dict[str, dict[str, set[int]]]:
    positives = interactions[
        (interactions["split"].isin(splits)) & (interactions["label"] == 1)
    ].sort_values(["user_id", "timestamp", "item_id"])
    profiles: dict[str, dict[str, set[int]]] = {}
    for user_id, group in positives.groupby("user_id"):
        category_ids: set[int] = set()
        tag_ids: set[int] = set()
        for item_id in group["item_id"].astype(str).tolist()[-max_history_len:]:
            item_info = item_lookup.get(item_id, {})
            category_ids.update(_nonzero_ints(item_info.get("category_indices", [])))
            tag_ids.update(_nonzero_ints(item_info.get("tag_indices", [])))
        profiles[str(user_id)] = {"categories": category_ids, "tags": tag_ids}
    return profiles


def _content_cross_features(
    history_profile: dict[str, set[int]],
    item_info: dict[str, Any],
) -> dict[str, float]:
    history_categories = history_profile.get("categories", set())
    history_tags = history_profile.get("tags", set())
    item_categories = set(_nonzero_ints(item_info.get("category_indices", [])))
    item_tags = set(_nonzero_ints(item_info.get("tag_indices", [])))
    category_overlap = len(history_categories & item_categories)
    tag_overlap = len(history_tags & item_tags)
    return {
        "category_overlap": float(category_overlap),
        "tag_overlap": float(tag_overlap),
        "category_jaccard": _jaccard(history_categories, item_categories),
        "tag_jaccard": _jaccard(history_tags, item_tags),
        "history_category_count": float(len(history_categories)),
        "history_tag_count": float(len(history_tags)),
    }


def _nonzero_ints(values: Any) -> list[int]:
    result: list[int] = []
    if values is None:
        return result
    for value in values:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric != 0:
            result.append(numeric)
    return result


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _history_indices(
    interactions: pd.DataFrame,
    splits: list[str],
    encoders: FeatureEncoders,
    max_history_len: int,
) -> dict[str, list[int]]:
    positives = interactions[
        (interactions["split"].isin(splits)) & (interactions["label"] == 1)
    ].sort_values(["user_id", "timestamp", "item_id"])
    histories: dict[str, list[int]] = {}
    for user_id, group in positives.groupby("user_id"):
        histories[str(user_id)] = [
            encoders.item.transform_one(item_id)
            for item_id in group["item_id"].astype(str)
        ][-max_history_len:]
    return histories


def _split_timestamps(interactions: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split, split_frame in interactions.groupby("split"):
        result[str(split)] = {
            str(user_id): int(group["timestamp"].max())
            for user_id, group in split_frame.groupby("user_id")
        }
    return result


def _filter_candidates(candidates: list[Candidate], seen: set[str]) -> list[Candidate]:
    if not seen:
        return candidates
    filtered = [candidate for candidate in candidates if candidate.item_id not in seen]
    for rank, candidate in enumerate(filtered, start=1):
        candidate.rank = rank
    return filtered


def _sources(candidate: Candidate) -> list[str]:
    if candidate.source == "":
        return []
    sources = [source for source in str(candidate.source or "unknown").split("+") if source]
    return sources or ["unknown"]


def _split(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_source_weights(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    weights: dict[str, float] = {}
    for part in _split(text):
        if "=" not in part:
            raise ValueError(f"Invalid source weight '{part}'. Expected name=value.")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


if __name__ == "__main__":
    main()
