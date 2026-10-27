from __future__ import annotations

import math
from collections import Counter

import numpy as np


def ranking_metrics(recommended: list[str], relevant: set[str], ks: list[int]) -> dict[str, float]:
    if len(recommended) != len(set(recommended)):
        raise ValueError("Recommendation lists must contain unique item IDs.")
    if any(k <= 0 for k in ks):
        raise ValueError("Metric cutoffs must be positive.")
    metrics: dict[str, float] = {}
    for k in ks:
        top = recommended[:k]
        hits = [1 if item_id in relevant else 0 for item_id in top]
        hit_count = sum(hits)
        metrics[f"Recall@{k}"] = hit_count / len(relevant) if relevant else 0.0
        metrics[f"Precision@{k}"] = hit_count / k if k else 0.0
        metrics[f"HitRate@{k}"] = 1.0 if hit_count > 0 else 0.0
        metrics[f"NDCG@{k}"] = _ndcg(hits, min(len(relevant), k))
    metrics["MRR@10"] = _mrr(recommended[:10], relevant)
    metrics["MAP@10"] = _average_precision(recommended[:10], relevant)
    return metrics


def catalog_coverage(recommended_by_user: dict[str, list[str]], all_items: set[str]) -> float:
    recommended = {item for items in recommended_by_user.values() for item in items}
    return len(recommended) / len(all_items) if all_items else 0.0


def item_coverage_count(recommended_by_user: dict[str, list[str]]) -> int:
    return len({item for items in recommended_by_user.values() for item in items})


def average_popularity(recommended_by_user: dict[str, list[str]], popularity: Counter[str]) -> float:
    values = [popularity.get(item, 0) for items in recommended_by_user.values() for item in items]
    return float(np.mean(values)) if values else 0.0


def tail_ratio(recommended_by_user: dict[str, list[str]], tail_items: set[str]) -> float:
    values = [item in tail_items for items in recommended_by_user.values() for item in items]
    return sum(values) / len(values) if values else 0.0


def category_diversity(
    recommended_by_user: dict[str, list[str]],
    item_meta: dict[str, dict[str, object]],
) -> float:
    values: list[float] = []
    for items in recommended_by_user.values():
        if not items:
            continue
        categories = []
        for item_id in items:
            raw = item_meta.get(item_id, {}).get("categories", [])
            if isinstance(raw, (list, tuple, set, np.ndarray)):
                categories.extend(str(category) for category in raw if str(category))
            elif raw is not None and str(raw):
                categories.append(str(raw))
        values.append(len(set(categories)) / max(1, len(items)))
    return float(np.mean(values)) if values else 0.0


def novelty(recommended_by_user: dict[str, list[str]], popularity: Counter[str]) -> float:
    values = [
        1.0 / math.log2(popularity.get(item, 0) + 2.0)
        for items in recommended_by_user.values()
        for item in items
    ]
    return float(np.mean(values)) if values else 0.0


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _mrr(recommended: list[str], relevant: set[str]) -> float:
    for idx, item_id in enumerate(recommended, start=1):
        if item_id in relevant:
            return 1.0 / idx
    return 0.0


def _average_precision(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant or not recommended:
        return 0.0
    hits = 0
    total = 0.0
    for idx, item_id in enumerate(recommended, start=1):
        if item_id in relevant:
            hits += 1
            total += hits / idx
    return total / min(len(relevant), k)
