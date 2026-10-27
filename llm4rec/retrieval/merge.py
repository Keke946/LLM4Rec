from __future__ import annotations

from collections import defaultdict

from llm4rec.retrieval.base import Candidate


def weighted_merge(
    candidate_lists: list[list[Candidate]],
    weights: dict[str, float] | None = None,
    top_k: int = 100,
) -> list[Candidate]:
    weights = weights or {}
    grouped: dict[str, dict[str, object]] = {}
    source_scores: dict[str, list[float]] = defaultdict(list)
    for candidates in candidate_lists:
        for candidate in candidates:
            source_scores[candidate.source].append(candidate.score)

    def normalize(candidate: Candidate) -> float:
        scores = source_scores[candidate.source]
        if not scores:
            return 0.0
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return 1.0
        return (candidate.score - min_score) / (max_score - min_score)

    for candidates in candidate_lists:
        for candidate in candidates:
            entry = grouped.setdefault(
                candidate.item_id,
                {"score": 0.0, "sources": set(), "best_rank": candidate.rank},
            )
            entry["score"] = float(entry["score"]) + weights.get(candidate.source, 1.0) * normalize(candidate)
            entry["sources"].add(candidate.source)
            entry["best_rank"] = min(int(entry["best_rank"]), candidate.rank)

    ranked = sorted(grouped.items(), key=lambda pair: (float(pair[1]["score"]), -int(pair[1]["best_rank"])), reverse=True)
    return [
        Candidate(
            item_id=item_id,
            score=float(info["score"]),
            source="+".join(sorted(info["sources"])),
            rank=rank,
            retrieval_score=float(info["score"]),
        )
        for rank, (item_id, info) in enumerate(ranked[:top_k], start=1)
    ]


def reciprocal_rank_fusion(
    candidate_lists: list[list[Candidate]],
    top_k: int = 100,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    for candidates in candidate_lists:
        for candidate in candidates:
            scores[candidate.item_id] += weights.get(candidate.source, 1.0) / (k + candidate.rank)
            sources[candidate.item_id].add(candidate.source)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        Candidate(
            item_id=item_id,
            score=float(score),
            source="+".join(sorted(sources[item_id])),
            rank=rank,
            retrieval_score=float(score),
        )
        for rank, (item_id, score) in enumerate(ranked[:top_k], start=1)
    ]
