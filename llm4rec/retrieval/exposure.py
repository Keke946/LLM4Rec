from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from llm4rec.retrieval.base import BaseRetriever, Candidate


class ExposureQualityRetriever(BaseRetriever):
    """Time-decayed exposure-quality retriever based on watch_ratio signals.

    This branch is intentionally different from positive-only popularity. It
    can surface items that have no train positives but have recent, high-quality
    exposures, which is common in short-video data with sparse positives.
    """

    name = "exposure"

    def __init__(
        self,
        half_life_days: float = 1.0,
        watch_ratio_power: float = 1.0,
        positive_boost: float = 0.5,
        score_floor: float = 0.001,
    ) -> None:
        super().__init__(
            half_life_days=half_life_days,
            watch_ratio_power=watch_ratio_power,
            positive_boost=positive_boost,
            score_floor=score_floor,
        )
        self.half_life_days = float(half_life_days)
        self.watch_ratio_power = float(watch_ratio_power)
        self.positive_boost = float(positive_boost)
        self.score_floor = float(score_floor)
        self.exposure_scores: dict[str, float] = {}
        self.global_ranking: list[tuple[str, float]] = []

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "ExposureQualityRetriever":
        super().fit(train_interactions, items)
        self.exposure_scores = self._build_exposure_scores(train_interactions)
        self.global_ranking = sorted(self.exposure_scores.items(), key=lambda pair: pair[1], reverse=True)
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        seen = self.train_positive_by_user.get(str(user_id), set())
        candidates: list[Candidate] = []
        rank = 1
        for item_id, score in self.global_ranking:
            if item_id in seen:
                continue
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

    def _build_exposure_scores(self, interactions: pd.DataFrame) -> dict[str, float]:
        if interactions.empty:
            return {str(item_id): self.score_floor for item_id in self.item_ids}
        max_ts = float(pd.to_numeric(interactions.get("timestamp", 0), errors="coerce").fillna(0).max())
        half_life_seconds = max(1.0, self.half_life_days * 86400.0)
        scores: dict[str, float] = defaultdict(float)
        timestamps = pd.to_numeric(interactions.get("timestamp", 0), errors="coerce").fillna(0)
        watch_ratios = pd.to_numeric(interactions.get("watch_ratio", 0), errors="coerce").fillna(0).clip(lower=0)
        labels = pd.to_numeric(interactions.get("label", 0), errors="coerce").fillna(0).clip(lower=0)
        for item_id, ts, watch_ratio, label in zip(
            interactions["item_id"].astype(str),
            timestamps,
            watch_ratios,
            labels,
        ):
            age = max(0.0, max_ts - float(ts))
            decay = 0.5 ** (age / half_life_seconds)
            quality = float(watch_ratio) ** self.watch_ratio_power
            score = decay * quality * (1.0 + self.positive_boost * float(label))
            if score > 0:
                scores[str(item_id)] += score
        max_score = max(scores.values()) if scores else 1.0
        normalized = {
            item_id: self.score_floor + float(score) / max_score
            for item_id, score in scores.items()
        }
        for item_id in self.item_ids:
            normalized.setdefault(str(item_id), self.score_floor)
        return normalized

    def _state_dict(self) -> dict[str, Any]:
        return {
            "exposure_scores": self.exposure_scores,
            "global_ranking": self.global_ranking,
        }

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        self.exposure_scores = {
            str(item_id): float(score)
            for item_id, score in state.get("exposure_scores", {}).items()
        }
        self.global_ranking = [
            (str(item_id), float(score))
            for item_id, score in state.get("global_ranking", [])
        ]
        if not self.global_ranking:
            self.global_ranking = sorted(self.exposure_scores.items(), key=lambda pair: pair[1], reverse=True)
