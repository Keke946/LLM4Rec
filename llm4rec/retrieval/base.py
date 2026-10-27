from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    item_id: str
    score: float
    source: str
    rank: int
    retrieval_score: float | None = None
    ranker_score: float | None = None


class BaseRetriever:
    name = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.items: pd.DataFrame | None = None
        self.train_interactions: pd.DataFrame | None = None
        self.train_positive_by_user: dict[str, set[str]] = {}
        self.item_ids: list[str] = []

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "BaseRetriever":
        self.items = items.copy()
        self.train_interactions = train_interactions.copy()
        self.item_ids = items["item_id"].astype(str).tolist() if "item_id" in items.columns else []
        positives = self.train_interactions[self.train_interactions["label"] == 1]
        self.train_positive_by_user = {
            str(user_id): set(group["item_id"].astype(str))
            for user_id, group in positives.groupby("user_id")
        }
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        target = artifact_dir(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "class": self.__class__.__name__,
                    "init_kwargs": self.init_kwargs(),
                    "format_version": 2,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "item_ids.json").write_text(
            json.dumps(self.item_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        train_seen = {
            user_id: sorted(items)
            for user_id, items in self.train_positive_by_user.items()
        }
        (target / "train_seen_by_user.json").write_text(
            json.dumps(train_seen, ensure_ascii=False),
            encoding="utf-8",
        )
        self._save_state(target)

    @classmethod
    def load(cls, path: str | Path) -> "BaseRetriever":
        target = artifact_dir(path)
        if (target / "config.json").exists():
            from llm4rec.retrieval import RETRIEVER_REGISTRY

            config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            retriever_cls = RETRIEVER_REGISTRY[config["name"]]
            retriever = retriever_cls(**config.get("init_kwargs", {}))
            item_ids_path = target / "item_ids.json"
            if item_ids_path.exists():
                retriever.item_ids = json.loads(item_ids_path.read_text(encoding="utf-8"))
            train_seen_path = target / "train_seen_by_user.json"
            if train_seen_path.exists():
                raw_seen = json.loads(train_seen_path.read_text(encoding="utf-8"))
                retriever.train_positive_by_user = {
                    str(user_id): set(map(str, item_ids))
                    for user_id, item_ids in raw_seen.items()
                }
            retriever._load_state(target)
            return retriever

        legacy = Path(path)
        if legacy.exists() and legacy.is_file():
            logger.warning("Loading legacy pickled retriever artifact: %s", legacy)
            with legacy.open("rb") as file:
                return pickle.load(file)
        raise FileNotFoundError(f"Retriever artifact not found: {path}")

    def init_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def _save_state(self, target: Path) -> None:
        with (target / "state.pkl").open("wb") as file:
            pickle.dump(self._state_dict(), file)

    def _load_state(self, target: Path) -> None:
        state_path = target / "state.pkl"
        if state_path.exists():
            with state_path.open("rb") as file:
                self._load_state_dict(pickle.load(file))

    def _state_dict(self) -> dict[str, Any]:
        return {}

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    def _filter_seen(self, user_id: str, scores: list[tuple[str, float]], top_k: int) -> list[Candidate]:
        seen = self.train_positive_by_user.get(str(user_id), set())
        candidates: list[Candidate] = []
        rank = 1
        for item_id, score in sorted(scores, key=lambda pair: pair[1], reverse=True):
            if item_id in seen:
                continue
            candidates.append(
                Candidate(
                    item_id=str(item_id),
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


def artifact_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix:
        return target.with_suffix("")
    return target
