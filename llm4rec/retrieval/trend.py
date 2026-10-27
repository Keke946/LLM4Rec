from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd

from llm4rec.datasets.base import normalize_list
from llm4rec.retrieval.base import BaseRetriever, Candidate


class TrendRetriever(BaseRetriever):
    """Time-decayed trend retriever with light user metadata affinity.

    Short-video recommendation often has strong freshness effects. A plain
    popularity retriever treats a positive from weeks ago the same as a positive
    from the newest training window. This retriever gives recent positives more
    weight and then lightly personalizes by the user's category/tag history.
    """

    name = "trend"

    def __init__(
        self,
        half_life_days: float = 2.0,
        category_weight: float = 0.4,
        tag_weight: float = 0.2,
        popularity_floor: float = 0.02,
        max_profile_items: int = 80,
        candidate_cache_k: int = 2000,
    ) -> None:
        super().__init__(
            half_life_days=half_life_days,
            category_weight=category_weight,
            tag_weight=tag_weight,
            popularity_floor=popularity_floor,
            max_profile_items=max_profile_items,
            candidate_cache_k=candidate_cache_k,
        )
        self.half_life_days = float(half_life_days)
        self.category_weight = float(category_weight)
        self.tag_weight = float(tag_weight)
        self.popularity_floor = float(popularity_floor)
        self.max_profile_items = int(max_profile_items)
        self.candidate_cache_k = int(candidate_cache_k)
        self.item_features: dict[str, dict[str, list[str]]] = {}
        self.trend_scores: dict[str, float] = {}
        self.user_profiles: dict[str, dict[str, dict[str, float]]] = {}
        self.user_rankings: dict[str, list[tuple[str, float]]] = {}
        self.global_ranking: list[tuple[str, float]] = []

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "TrendRetriever":
        super().fit(train_interactions, items)
        self.item_features = self._build_item_features(items)
        positives = train_interactions[train_interactions["label"] == 1].copy()
        self.trend_scores = self._build_trend_scores(positives)
        self.user_profiles = self._build_user_profiles(positives)
        self.global_ranking = sorted(
            ((str(item_id), float(self.trend_scores.get(str(item_id), self.popularity_floor))) for item_id in self.item_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )
        personalized_item_ids = [item_id for item_id, _ in self.global_ranking[: self.candidate_cache_k]]
        self.user_rankings = {
            user_id: self._rank_for_profile(profile, personalized_item_ids)
            for user_id, profile in self.user_profiles.items()
        }
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        user_id = str(user_id)
        ranking = self.user_rankings.get(user_id) or self.global_ranking
        fallback = None if ranking is self.global_ranking else self.global_ranking
        return self._filter_seen_sorted(user_id, ranking, top_k, fallback=fallback)

    def _build_item_features(self, items: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
        result: dict[str, dict[str, list[str]]] = {}
        for _, row in items.iterrows():
            result[str(row["item_id"])] = {
                "categories": normalize_list(row.get("categories", [])),
                "tags": normalize_list(row.get("tags", [])),
            }
        return result

    def _build_trend_scores(self, positives: pd.DataFrame) -> dict[str, float]:
        if positives.empty or "timestamp" not in positives.columns:
            return {item_id: self.popularity_floor for item_id in self.item_ids}
        max_ts = float(pd.to_numeric(positives["timestamp"], errors="coerce").fillna(0).max())
        half_life_seconds = max(1.0, self.half_life_days * 86400.0)
        scores: dict[str, float] = defaultdict(float)
        for _, row in positives.iterrows():
            item_id = str(row["item_id"])
            ts = float(row.get("timestamp", 0) or 0)
            age = max(0.0, max_ts - ts)
            scores[item_id] += 0.5 ** (age / half_life_seconds)
        max_score = max(scores.values()) if scores else 1.0
        normalized = {
            item_id: self.popularity_floor + float(score) / max_score
            for item_id, score in scores.items()
        }
        for item_id in self.item_ids:
            normalized.setdefault(str(item_id), self.popularity_floor)
        return normalized

    def _build_user_profiles(self, positives: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
        profiles: dict[str, dict[str, dict[str, float]]] = {}
        if positives.empty:
            return profiles
        positives = positives.sort_values(["user_id", "timestamp", "item_id"])
        for user_id, group in positives.groupby("user_id"):
            history = group["item_id"].astype(str).tolist()[-self.max_profile_items :]
            categories: dict[str, float] = defaultdict(float)
            tags: dict[str, float] = defaultdict(float)
            total = len(history)
            for idx, item_id in enumerate(history):
                weight = (idx + 1) / max(1, total)
                features = self.item_features.get(item_id, {})
                for category in features.get("categories", []):
                    categories[str(category)] += weight
                for tag in features.get("tags", []):
                    tags[str(tag)] += weight
            profiles[str(user_id)] = {"categories": dict(categories), "tags": dict(tags)}
        return profiles

    def _affinity(self, profile: dict[str, dict[str, float]], features: dict[str, list[str]]) -> float:
        category_score = sum(
            profile.get("categories", {}).get(str(category), 0.0)
            for category in features.get("categories", [])
        )
        tag_score = sum(
            profile.get("tags", {}).get(str(tag), 0.0)
            for tag in features.get("tags", [])
        )
        norm = 1.0 + sum(profile.get("categories", {}).values()) + sum(profile.get("tags", {}).values())
        return (self.category_weight * category_score + self.tag_weight * tag_score) / norm

    def _rank_for_profile(
        self,
        profile: dict[str, dict[str, float]],
        item_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for item_id in item_ids or self.item_ids:
            item_id = str(item_id)
            base = self.trend_scores.get(item_id, self.popularity_floor)
            score = base * (1.0 + self._affinity(profile, self.item_features.get(item_id, {})))
            if score > 0:
                scores.append((item_id, float(score)))
        return sorted(scores, key=lambda pair: pair[1], reverse=True)

    def _filter_seen_sorted(
        self,
        user_id: str,
        ranking: list[tuple[str, float]],
        top_k: int,
        fallback: list[tuple[str, float]] | None = None,
    ) -> list[Candidate]:
        seen = self.train_positive_by_user.get(str(user_id), set())
        candidates: list[Candidate] = []
        emitted: set[str] = set()
        rank = 1
        for item_id, score in _chain_rankings(ranking, fallback):
            item_id = str(item_id)
            if item_id in emitted or item_id in seen:
                continue
            emitted.add(item_id)
            candidates.append(
                Candidate(
                    item_id=item_id,
                    score=float(score),
                    source=self.name,
                    rank=rank,
                    retrieval_score=float(score),
                )
            )
            rank += 1
            if len(candidates) >= top_k:
                break
        return candidates

    def _state_dict(self) -> dict[str, Any]:
        return {
            "item_features": self.item_features,
            "trend_scores": self.trend_scores,
            "user_profiles": self.user_profiles,
            "user_rankings": self.user_rankings,
            "global_ranking": self.global_ranking,
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self.item_features = {
            str(item_id): {
                "categories": normalize_list(features.get("categories", [])),
                "tags": normalize_list(features.get("tags", [])),
            }
            for item_id, features in state.get("item_features", {}).items()
        }
        self.trend_scores = {
            str(item_id): float(score)
            for item_id, score in state.get("trend_scores", {}).items()
        }
        self.user_profiles = {
            str(user_id): {
                str(kind): {str(token): float(weight) for token, weight in values.items()}
                for kind, values in profile.items()
            }
            for user_id, profile in state.get("user_profiles", {}).items()
        }
        self.user_rankings = {
            str(user_id): [(str(item_id), float(score)) for item_id, score in ranking]
            for user_id, ranking in state.get("user_rankings", {}).items()
        }
        self.global_ranking = [
            (str(item_id), float(score))
            for item_id, score in state.get("global_ranking", [])
        ]
        if not self.global_ranking:
            self.global_ranking = sorted(self.trend_scores.items(), key=lambda pair: pair[1], reverse=True)
        if not self.user_rankings and self.user_profiles:
            personalized_item_ids = [item_id for item_id, _ in self.global_ranking[: self.candidate_cache_k]]
            self.user_rankings = {
                user_id: self._rank_for_profile(profile, personalized_item_ids)
                for user_id, profile in self.user_profiles.items()
            }


def _chain_rankings(
    primary: list[tuple[str, float]],
    fallback: list[tuple[str, float]] | None,
) -> list[tuple[str, float]]:
    if not fallback:
        return primary
    return [*primary, *fallback]
