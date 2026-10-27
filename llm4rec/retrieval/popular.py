from __future__ import annotations

import pandas as pd

from llm4rec.retrieval.base import BaseRetriever, Candidate


class PopularRetriever(BaseRetriever):
    name = "popular"

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "PopularRetriever":
        super().fit(train_interactions, items)
        positives = train_interactions[train_interactions["label"] == 1]
        counts = positives.groupby("item_id").size().sort_values(ascending=False)
        self.popularity_scores = {str(item_id): float(score) for item_id, score in counts.items()}
        for item_id in items["item_id"].astype(str):
            self.popularity_scores.setdefault(item_id, 0.0)
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        return self._filter_seen(str(user_id), list(self.popularity_scores.items()), top_k)

    def _state_dict(self) -> dict:
        return {"popularity_scores": self.popularity_scores}

    def _load_state_dict(self, state: dict) -> None:
        self.popularity_scores = {
            str(item_id): float(score)
            for item_id, score in state.get("popularity_scores", {}).items()
        }
