from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from llm4rec.retrieval.base import BaseRetriever, Candidate


class ItemCFRetriever(BaseRetriever):
    name = "itemcf"

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "ItemCFRetriever":
        super().fit(train_interactions, items)
        positives = train_interactions[train_interactions["label"] == 1]
        co_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        item_counts: dict[str, int] = defaultdict(int)

        for _, group in positives.groupby("user_id"):
            item_ids = sorted(set(group["item_id"].astype(str)))
            for item_id in item_ids:
                item_counts[item_id] += 1
            weight = 1.0 / math.log2(len(item_ids) + 1) if len(item_ids) > 1 else 1.0
            for i, item_i in enumerate(item_ids):
                for item_j in item_ids[i + 1 :]:
                    co_counts[item_i][item_j] += weight
                    co_counts[item_j][item_i] += weight

        self.similar_items: dict[str, list[tuple[str, float]]] = {}
        for item_i, related in co_counts.items():
            sims = []
            for item_j, count in related.items():
                denom = math.sqrt(item_counts[item_i] * item_counts[item_j])
                sims.append((item_j, count / denom if denom else 0.0))
            self.similar_items[item_i] = sorted(sims, key=lambda pair: pair[1], reverse=True)
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        history = self.train_positive_by_user.get(str(user_id), set())
        scores: dict[str, float] = defaultdict(float)
        for item_id in sorted(history):
            for related_item, sim in self.similar_items.get(item_id, [])[:200]:
                scores[related_item] += sim
        return self._filter_seen(str(user_id), list(scores.items()), top_k)

    def _state_dict(self) -> dict:
        return {"similar_items": self.similar_items}

    def _load_state_dict(self, state: dict) -> None:
        self.similar_items = {
            str(item_id): [(str(related), float(score)) for related, score in related_items]
            for item_id, related_items in state.get("similar_items", {}).items()
        }
