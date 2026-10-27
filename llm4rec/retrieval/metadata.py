from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd

from llm4rec.datasets.base import normalize_list
from llm4rec.retrieval.base import BaseRetriever, Candidate


class MetadataRetriever(BaseRetriever):
    """Tail-aware metadata retriever built from user category/tag/author profiles.

    This branch gives the recall stage a non-collaborative path for low-popularity
    items. It is intentionally simple and version-safe: all saved state is made
    from primitive dict/list values rather than pandas objects.
    """

    name = "metadata"

    def __init__(
        self,
        category_weight: float = 2.0,
        tag_weight: float = 1.0,
        author_weight: float = 0.5,
        popularity_weight: float = 0.08,
        tail_boost: float = 0.35,
        recency_decay: float = 0.97,
        max_profile_items: int = 80,
    ) -> None:
        super().__init__(
            category_weight=category_weight,
            tag_weight=tag_weight,
            author_weight=author_weight,
            popularity_weight=popularity_weight,
            tail_boost=tail_boost,
            recency_decay=recency_decay,
            max_profile_items=max_profile_items,
        )
        self.category_weight = float(category_weight)
        self.tag_weight = float(tag_weight)
        self.author_weight = float(author_weight)
        self.popularity_weight = float(popularity_weight)
        self.tail_boost = float(tail_boost)
        self.recency_decay = float(recency_decay)
        self.max_profile_items = int(max_profile_items)
        self.item_features: dict[str, dict[str, Any]] = {}
        self.item_popularity: dict[str, float] = {}
        self.user_profiles: dict[str, dict[str, dict[str, float]]] = {}
        self.global_scores: dict[str, float] = {}

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "MetadataRetriever":
        super().fit(train_interactions, items)
        self.item_features = self._build_item_features(items)

        positives = train_interactions[train_interactions["label"] == 1].copy()
        self.item_popularity = {
            str(item_id): float(count)
            for item_id, count in positives.groupby("item_id").size().items()
        }
        for item_id in self.item_ids:
            self.item_popularity.setdefault(str(item_id), 0.0)

        self.global_scores = {
            item_id: self._global_item_score(item_id)
            for item_id in self.item_ids
        }
        self.user_profiles = self._build_user_profiles(positives)
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        user_id = str(user_id)
        profile = self.user_profiles.get(user_id)
        if not profile:
            return self._filter_seen(user_id, list(self.global_scores.items()), top_k)

        scores: list[tuple[str, float]] = []
        for item_id in self.item_ids:
            features = self.item_features.get(str(item_id), {})
            score = self._profile_score(profile, features) + self.global_scores.get(str(item_id), 0.0)
            if score > 0:
                scores.append((str(item_id), float(score)))
        return self._filter_seen(user_id, scores, top_k)

    def _build_item_features(self, items: pd.DataFrame) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {}
        for _, row in items.iterrows():
            item_id = str(row["item_id"])
            categories = normalize_list(row.get("categories", []))
            tags = normalize_list(row.get("tags", []))
            features[item_id] = {
                "categories": categories,
                "tags": tags,
                "author_id": str(row.get("author_id", "") or ""),
            }
        return features

    def _build_user_profiles(self, positives: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
        profiles: dict[str, dict[str, dict[str, float]]] = {}
        if positives.empty:
            return profiles

        positives = positives.sort_values(["user_id", "timestamp", "item_id"])
        for user_id, group in positives.groupby("user_id"):
            history = group["item_id"].astype(str).tolist()[-self.max_profile_items :]
            profile = {
                "categories": defaultdict(float),
                "tags": defaultdict(float),
                "authors": defaultdict(float),
            }
            total = len(history)
            for idx, item_id in enumerate(history):
                weight = self.recency_decay ** max(0, total - idx - 1)
                features = self.item_features.get(item_id, {})
                for category in features.get("categories", []):
                    profile["categories"][str(category)] += weight
                for tag in features.get("tags", []):
                    profile["tags"][str(tag)] += weight
                author_id = str(features.get("author_id", "") or "")
                if author_id:
                    profile["authors"][author_id] += weight
            profiles[str(user_id)] = {
                key: dict(values)
                for key, values in profile.items()
            }
        return profiles

    def _profile_score(self, profile: dict[str, dict[str, float]], features: dict[str, Any]) -> float:
        category_score = sum(
            profile.get("categories", {}).get(str(category), 0.0)
            for category in features.get("categories", [])
        )
        tag_score = sum(
            profile.get("tags", {}).get(str(tag), 0.0)
            for tag in features.get("tags", [])
        )
        author = str(features.get("author_id", "") or "")
        author_score = profile.get("authors", {}).get(author, 0.0) if author else 0.0
        return (
            self.category_weight * category_score
            + self.tag_weight * tag_score
            + self.author_weight * author_score
        )

    def _global_item_score(self, item_id: str) -> float:
        popularity = self.item_popularity.get(str(item_id), 0.0)
        popularity_score = self.popularity_weight * math.log1p(popularity)
        tail_score = self.tail_boost / math.sqrt(popularity + 1.0)
        return popularity_score + tail_score

    def _state_dict(self) -> dict[str, Any]:
        return {
            "item_features": self.item_features,
            "item_popularity": self.item_popularity,
            "user_profiles": self.user_profiles,
            "global_scores": self.global_scores,
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self.item_features = {
            str(item_id): {
                "categories": normalize_list(features.get("categories", [])),
                "tags": normalize_list(features.get("tags", [])),
                "author_id": str(features.get("author_id", "") or ""),
            }
            for item_id, features in state.get("item_features", {}).items()
        }
        self.item_popularity = {
            str(item_id): float(score)
            for item_id, score in state.get("item_popularity", {}).items()
        }
        self.user_profiles = {
            str(user_id): {
                str(kind): {str(token): float(weight) for token, weight in values.items()}
                for kind, values in profile.items()
            }
            for user_id, profile in state.get("user_profiles", {}).items()
        }
        self.global_scores = {
            str(item_id): float(score)
            for item_id, score in state.get("global_scores", {}).items()
        }
