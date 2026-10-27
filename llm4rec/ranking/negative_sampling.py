from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_NEGATIVE_MIX = {
    "random_negative": 0.2,
    "exposed_low_watch_negative": 0.4,
    "retrieved_hard_negative": 0.3,
    "popularity_hard_negative": 0.1,
}

NEGATIVE_TYPES = tuple(DEFAULT_NEGATIVE_MIX)


def positive_items_by_user(
    interactions: pd.DataFrame,
    splits: list[str] | None = None,
    label_col: str = "label",
) -> dict[str, set[str]]:
    frame = interactions
    if splits is not None and "split" in frame.columns:
        frame = frame[frame["split"].isin(splits)]
    if label_col in frame.columns:
        frame = frame[pd.to_numeric(frame[label_col], errors="coerce").fillna(0) > 0]
    return {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in frame.groupby("user_id")
    }


def low_watch_items_by_user(
    interactions: pd.DataFrame,
    threshold: float = 0.3,
    allowed_splits: list[str] | None = None,
) -> dict[str, set[str]]:
    if "watch_ratio" not in interactions.columns:
        return {}
    frame = interactions.copy()
    if allowed_splits is not None and "split" in frame.columns:
        frame = frame[frame["split"].isin(allowed_splits)]
    ratios = pd.to_numeric(frame["watch_ratio"], errors="coerce")
    label_values = frame["label"] if "label" in frame.columns else pd.Series(0, index=frame.index)
    labels = pd.to_numeric(label_values, errors="coerce").fillna(0)
    frame = frame[(ratios < threshold) & (labels <= 0)]
    return {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in frame.groupby("user_id")
    }


def annotate_negative_types(
    frame: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    low_watch_threshold: float = 0.3,
    popularity_quantile: float = 0.8,
    future_positive_by_user: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    annotated = frame.copy()
    if annotated.empty:
        annotated["negative_type"] = []
        return annotated
    future_positive_by_user = future_positive_by_user or {}
    low_watch_by_user = (
        low_watch_items_by_user(interactions, threshold=low_watch_threshold, allowed_splits=["train", "prefix"])
        if interactions is not None
        else {}
    )
    popularity_values = (
        annotated["item_popularity"] if "item_popularity" in annotated.columns else pd.Series(0, index=annotated.index)
    )
    popularity = pd.to_numeric(popularity_values, errors="coerce").fillna(0.0)
    live_mask = (pd.to_numeric(annotated["is_injected_positive"], errors="coerce").fillna(0) <= 0
                 if "is_injected_positive" in annotated else pd.Series(True, index=annotated.index))
    live_popularity = popularity[live_mask]
    popularity_threshold = float(live_popularity.quantile(popularity_quantile)) if len(live_popularity) else 0.0

    user_ids = annotated["user_id"].astype(str)
    item_ids = annotated["item_id"].astype(str)
    labels = pd.to_numeric(annotated["label"], errors="coerce").fillna(0)
    pairs = pd.MultiIndex.from_arrays([user_ids, item_ids])
    low_pairs = {(user, item) for user, items in low_watch_by_user.items() for item in items}
    future_pairs = {(user, item) for user, items in future_positive_by_user.items() for item in items}
    low_mask = pairs.isin(low_pairs)
    future_mask = pairs.isin(future_pairs)
    sources = annotated.get("retrieval_source", annotated.get("source", pd.Series("", index=annotated.index)))
    source_mask = sources.fillna("").astype(str).str.len() > 0
    annotated["negative_type"] = np.select(
        [labels > 0, low_mask, (popularity >= popularity_threshold) & (popularity_threshold > 0), source_mask],
        ["positive", "exposed_low_watch_negative", "popularity_hard_negative", "retrieved_hard_negative"],
        default="random_negative",
    )
    annotated["_future_positive_leak"] = (labels <= 0) & future_mask
    return annotated


def sample_exposure_aware_negatives(
    frame: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    strategy: str = "mixed_negative",
    mix: dict[str, float] | None = None,
    negatives_per_positive: int | None = 20,
    seed: int = 42,
    include_injected_positives: bool = False,
    low_watch_threshold: float = 0.3,
    popularity_quantile: float = 0.8,
    future_positive_splits: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), _sampling_stats(frame.copy(), frame.copy())
    if future_positive_splits:
        raise ValueError("Future validation/test labels cannot select training negatives.")
    future_positive_by_user = {}
    # Negative typing may inspect only interactions available before the ranker
    # target window. The caller must supply the prefix, not full train/val/test.
    if interactions is not None:
        if "split" in interactions and not interactions["split"].isin(["train", "prefix"]).all():
            raise ValueError("Pass prefix-only interactions for exposure-aware sampling.")
    annotated = annotate_negative_types(
        frame,
        interactions=interactions,
        low_watch_threshold=low_watch_threshold,
        popularity_quantile=popularity_quantile,
        future_positive_by_user=future_positive_by_user,
    )
    positives = annotated[pd.to_numeric(annotated["label"], errors="coerce").fillna(0) > 0].copy()
    if not include_injected_positives and "is_injected_positive" in positives.columns:
        positives = positives[pd.to_numeric(positives["is_injected_positive"], errors="coerce").fillna(0) <= 0]
    negatives = annotated[
        (pd.to_numeric(annotated["label"], errors="coerce").fillna(0) <= 0)
        & (~annotated["_future_positive_leak"].astype(bool))
    ].copy()

    if strategy == "mixed":
        strategy = "mixed_negative"
    if strategy == "mixed_negative":
        sampled_negatives = _sample_mixed(negatives, len(positives), mix or DEFAULT_NEGATIVE_MIX, negatives_per_positive, seed)
    elif strategy == "random_negative":
        sampled_negatives = _sample_pool(negatives, len(positives), negatives_per_positive, seed)
        sampled_negatives = sampled_negatives.copy()
        sampled_negatives["negative_type"] = "random_negative"
    elif strategy in NEGATIVE_TYPES:
        pool = negatives[negatives["negative_type"] == strategy]
        if pool.empty:
            raise ValueError(f"No eligible negatives for {strategy}; refusing to substitute a different strategy.")
        sampled_negatives = _sample_pool(pool, len(positives), negatives_per_positive, seed)
    else:
        raise ValueError(f"Unknown negative sampling strategy: {strategy}")

    sampled = pd.concat([positives, sampled_negatives], ignore_index=True)
    sampled = sampled.drop(columns=["_future_positive_leak"], errors="ignore")
    sampled = sampled.sort_values(["user_id", "label", "rank", "item_id"], ascending=[True, False, True, True]).reset_index(drop=True)
    stats = _sampling_stats(annotated, sampled)
    stats["strategy"] = strategy
    stats["empty_pool_fallback"] = False
    stats["negative_pool_semantics"] = "random within live retrieval pool; hard pools use prefix-only exposure/popularity tags"
    stats["negative_mix"] = dict(mix or DEFAULT_NEGATIVE_MIX) if strategy == "mixed_negative" else {}
    stats["excluded_future_positive_negatives"] = int(annotated["_future_positive_leak"].sum())
    if "is_injected_positive" in annotated.columns:
        injected_positive_rows = annotated[
            (pd.to_numeric(annotated["label"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(annotated["is_injected_positive"], errors="coerce").fillna(0) > 0)
        ]
        stats["input_injected_positives"] = int(len(injected_positive_rows))
        stats["excluded_injected_positives"] = int(0 if include_injected_positives else len(injected_positive_rows))
    return sampled, stats


def stats_to_markdown(stats: dict[str, Any]) -> str:
    rows = [
        ("strategy", stats.get("strategy", "")),
        ("num_users", stats.get("num_users", 0)),
        ("num_candidates", stats.get("num_candidates", 0)),
        ("num_positives", stats.get("num_positives", 0)),
        ("num_negatives", stats.get("num_negatives", 0)),
        ("positive_ratio", _fmt(stats.get("positive_ratio", 0.0))),
        ("avg_candidates_per_user", _fmt(stats.get("avg_candidates_per_user", 0.0))),
        ("avg_positives_per_user", _fmt(stats.get("avg_positives_per_user", 0.0))),
        ("avg_negatives_per_user", _fmt(stats.get("avg_negatives_per_user", 0.0))),
        ("candidate_positive_recall", _fmt(stats.get("candidate_positive_recall", 0.0))),
        ("positive_injection_rate", _fmt(stats.get("positive_injection_rate", 0.0))),
        ("excluded_future_positive_negatives", stats.get("excluded_future_positive_negatives", 0)),
        ("excluded_injected_positives", stats.get("excluded_injected_positives", 0)),
    ]
    distribution = stats.get("negative_type_distribution", {})
    text = ["# Negative Sampling Stats", "", "| Metric | Value |", "|---|---:|"]
    text.extend(f"| {key} | {value} |" for key, value in rows)
    text.extend(["", "## Negative Type Distribution", "", "| negative_type | count |", "|---|---:|"])
    for key, value in sorted(distribution.items()):
        text.append(f"| {key} | {value} |")
    return "\n".join(text) + "\n"


def _sample_mixed(
    negatives: pd.DataFrame,
    positive_count: int,
    mix: dict[str, float],
    negatives_per_positive: int | None,
    seed: int,
) -> pd.DataFrame:
    target_total = _target_negative_count(len(negatives), positive_count, negatives_per_positive)
    if target_total >= len(negatives):
        return negatives.copy()
    frames = []
    used_indices: set[int] = set()
    normalized = _normalize_mix(mix)
    for idx, (negative_type, weight) in enumerate(normalized.items()):
        target = int(round(target_total * weight))
        if idx == len(normalized) - 1:
            target = target_total - sum(len(frame) for frame in frames)
        pool = negatives[(negatives["negative_type"] == negative_type) & (~negatives.index.isin(used_indices))]
        sampled = _sample_n(pool, target, seed + idx)
        used_indices.update(int(index) for index in sampled.index)
        frames.append(sampled)
    remaining = target_total - sum(len(frame) for frame in frames)
    if remaining > 0:
        pool = negatives[~negatives.index.isin(used_indices)]
        frames.append(_sample_n(pool, remaining, seed + 101))
    return pd.concat(frames, ignore_index=False) if frames else negatives.head(0).copy()


def _sample_pool(
    pool: pd.DataFrame,
    positive_count: int,
    negatives_per_positive: int | None,
    seed: int,
) -> pd.DataFrame:
    target = _target_negative_count(len(pool), positive_count, negatives_per_positive)
    return _sample_n(pool, target, seed)


def _sample_n(pool: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or pool.empty:
        return pool.head(0).copy()
    if count >= len(pool):
        return pool.copy()
    return pool.sample(n=count, random_state=seed)


def _target_negative_count(pool_size: int, positive_count: int, negatives_per_positive: int | None) -> int:
    if negatives_per_positive is None or negatives_per_positive <= 0 or positive_count <= 0:
        return pool_size
    return min(pool_size, int(positive_count * negatives_per_positive))


def _normalize_mix(mix: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value)) for key, value in mix.items() if key in NEGATIVE_TYPES}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_NEGATIVE_MIX
    return {key: value / total for key, value in cleaned.items()}


def _sampling_stats(input_frame: pd.DataFrame, sampled: pd.DataFrame) -> dict[str, Any]:
    users = int(sampled["user_id"].nunique()) if "user_id" in sampled.columns and not sampled.empty else 0
    label_values = sampled["label"] if "label" in sampled.columns else pd.Series(0, index=sampled.index)
    labels = pd.to_numeric(label_values, errors="coerce").fillna(0)
    positives = int((labels > 0).sum())
    negatives = int((labels <= 0).sum())
    if "negative_type" in sampled.columns:
        distribution = dict(Counter(sampled[labels <= 0]["negative_type"].astype(str)))
    else:
        distribution = {}
    input_labels = pd.to_numeric(input_frame.get("label", pd.Series(dtype=float)), errors="coerce").fillna(0)
    input_positives = int((input_labels > 0).sum())
    retrieved = int(input_frame.get("is_retrieved_positive", pd.Series(dtype=float)).sum()) if len(input_frame) else 0
    injected = int(input_frame.get("is_injected_positive", pd.Series(dtype=float)).sum()) if len(input_frame) else 0
    per_user = sampled.groupby("user_id") if "user_id" in sampled.columns and not sampled.empty else None
    return {
        "num_users": users,
        "num_candidates": int(len(sampled)),
        "num_input_candidates": int(len(input_frame)),
        "num_positives": positives,
        "num_negatives": negatives,
        "positive_ratio": float(positives / len(sampled)) if len(sampled) else 0.0,
        "negative_type_distribution": distribution,
        "avg_candidates_per_user": float(len(sampled) / users) if users else 0.0,
        "avg_positives_per_user": float(positives / users) if users else 0.0,
        "avg_negatives_per_user": float(negatives / users) if users else 0.0,
        "candidate_positive_recall": float(retrieved / input_positives) if input_positives else 0.0,
        "candidate_recall_denominator": "all input positives before exclusion and sampling",
        "positive_injection_rate": float(injected / input_positives) if input_positives else 0.0,
        "users_with_samples": int(len(per_user)) if per_user is not None else 0,
    }


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
