from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def compute_final_scores(
    retrieval_scores: list[float],
    ranker_logits: list[float],
    mode: str = "retrieval_ranker_fusion",
    retrieval_weight: float = 0.7,
    ranker_weight: float = 0.3,
) -> list[float]:
    if mode == "retrieval_only":
        return normalize_scores(retrieval_scores)
    ranker_probs = [sigmoid(value) for value in ranker_logits]
    if mode == "ranker_only":
        return ranker_probs
    normalized_retrieval = normalize_scores(retrieval_scores)
    return [
        retrieval_weight * retrieval_score + ranker_weight * ranker_prob
        for retrieval_score, ranker_prob in zip(normalized_retrieval, ranker_probs)
    ]
