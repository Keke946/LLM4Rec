from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm4rec.retrieval.base import BaseRetriever, Candidate


class ContentFaissRetriever(BaseRetriever):
    name = "content_faiss"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_profile_items: int = 5,
        max_profile_chars: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_profile_items = max_profile_items
        self.max_profile_chars = max_profile_chars
        self.backend = "tfidf"
        self.item_ids: list[str] = []
        self.item_texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self.user_profile_embeddings: dict[str, np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("backend") == "faiss" and state.get("index") is not None:
            try:
                import faiss

                state["_faiss_index_bytes"] = faiss.serialize_index(state["index"])
                state["index"] = None
            except Exception:
                state["_faiss_index_bytes"] = None
        return state

    def __setstate__(self, state):
        index_bytes = state.pop("_faiss_index_bytes", None)
        self.__dict__.update(state)
        if self.backend == "faiss" and index_bytes is not None:
            try:
                import faiss

                self.index = faiss.deserialize_index(index_bytes)
            except Exception:
                self.backend = "sentence_transformer_numpy"

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "ContentFaissRetriever":
        super().fit(train_interactions, items)
        self.item_ids = items["item_id"].astype(str).tolist()
        self.item_texts = items["full_text"].fillna("").astype(str).tolist()

        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.model_name)
            self.embeddings = np.asarray(
                self.embedding_model.encode(self.item_texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            try:
                import faiss

                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index.add(self.embeddings)
                self.backend = "faiss"
            except ImportError:
                self.backend = "sentence_transformer_numpy"
            self._build_user_profile_embeddings()
        except Exception:
            self.vectorizer = TfidfVectorizer(max_features=50000)
            self.matrix = self.vectorizer.fit_transform(self.item_texts)
            self.backend = "tfidf"
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        query_text = query or self._profile_text(str(user_id))
        if not query_text:
            query_text = " ".join(self.item_texts[:3])

        if self.backend == "faiss":
            query_vec = self._query_embedding(str(user_id), query, query_text)
            _, indices = self.index.search(query_vec, min(len(self.item_ids), top_k * 5))
            scores = [(self.item_ids[idx], float(top_k * 5 - rank)) for rank, idx in enumerate(indices[0]) if idx >= 0]
        elif self.backend == "sentence_transformer_numpy":
            query_vec = self._query_embedding(str(user_id), query, query_text)
            sims = np.dot(self.embeddings, query_vec.reshape(-1))
            order = np.argsort(-sims)[: min(len(self.item_ids), top_k * 5)]
            scores = [(self.item_ids[idx], float(sims[idx])) for idx in order]
        else:
            query_matrix = self.vectorizer.transform([query_text])
            sims = cosine_similarity(query_matrix, self.matrix).reshape(-1)
            order = np.argsort(-sims)[: min(len(self.item_ids), top_k * 5)]
            scores = [(self.item_ids[idx], float(sims[idx])) for idx in order]
        return self._filter_seen(str(user_id), scores, top_k)

    def _profile_text(self, user_id: str) -> str:
        positives = self.train_positive_by_user.get(user_id, set())
        if not positives:
            return ""
        item_text = dict(zip(self.item_ids, self.item_texts))
        return self._trim_profile_text(
            " ".join(str(item_text.get(item_id, "")) for item_id in sorted(positives)[: self.max_profile_items])
        )

    def _build_user_profile_embeddings(self) -> None:
        if self.embedding_model is None or self.items is None:
            return
        item_text = self.items.set_index("item_id")["full_text"].to_dict()
        user_ids: list[str] = []
        texts: list[str] = []
        for user_id, positives in sorted(self.train_positive_by_user.items()):
            text = self._trim_profile_text(
                " ".join(
                    str(item_text.get(item_id, ""))
                    for item_id in sorted(positives)[: self.max_profile_items]
                )
            )
            if text.strip():
                user_ids.append(str(user_id))
                texts.append(text)
        if not texts:
            self.user_profile_embeddings = {}
            return
        embeddings = np.asarray(
            self.embedding_model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )
        self.user_profile_embeddings = {
            user_id: embeddings[index]
            for index, user_id in enumerate(user_ids)
        }

    def _query_embedding(self, user_id: str, query: str | None, query_text: str) -> np.ndarray:
        cached = None if query else self.user_profile_embeddings.get(str(user_id))
        if cached is not None:
            return np.asarray(cached, dtype=np.float32).reshape(1, -1)
        return np.asarray(
            self.embedding_model.encode([query_text], normalize_embeddings=True),
            dtype=np.float32,
        )

    def _trim_profile_text(self, text: str) -> str:
        if self.max_profile_chars <= 0:
            return text
        return text[: self.max_profile_chars]

    def init_kwargs(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_profile_items": self.max_profile_items,
            "max_profile_chars": self.max_profile_chars,
        }

    def _save_state(self, target: Path) -> None:
        state = {
            "model_name": self.model_name,
            "max_profile_items": self.max_profile_items,
            "max_profile_chars": self.max_profile_chars,
            "backend": "tfidf" if self.backend == "tfidf" else "sentence_transformer_numpy",
            "item_ids": self.item_ids,
            "item_texts": self.item_texts,
            "user_profile_ids": list(self.user_profile_embeddings),
            "vectorizer": self.vectorizer if self.backend == "tfidf" else None,
            "matrix": self.matrix if self.backend == "tfidf" else None,
        }
        with (target / "state.pkl").open("wb") as file:
            pickle.dump(state, file)
        if self.embeddings is not None:
            profile_ids = list(self.user_profile_embeddings)
            profile_embeddings = (
                np.asarray([self.user_profile_embeddings[user_id] for user_id in profile_ids], dtype=np.float32)
                if profile_ids
                else np.empty((0, self.embeddings.shape[1]), dtype=np.float32)
            )
            np.savez_compressed(
                target / "state.npz",
                embeddings=self.embeddings,
                user_profile_embeddings=profile_embeddings,
            )

    def _load_state(self, target: Path) -> None:
        with (target / "state.pkl").open("rb") as file:
            state = pickle.load(file)
        self.model_name = state.get("model_name", self.model_name)
        self.max_profile_items = int(state.get("max_profile_items", self.max_profile_items))
        self.max_profile_chars = int(state.get("max_profile_chars", self.max_profile_chars))
        self.backend = state.get("backend", "tfidf")
        self.item_ids = [str(item_id) for item_id in state.get("item_ids", [])]
        self.item_texts = [str(text) for text in state.get("item_texts", [])]
        user_profile_ids = [str(user_id) for user_id in state.get("user_profile_ids", [])]
        self.vectorizer = state.get("vectorizer")
        self.matrix = state.get("matrix")
        self.user_profile_embeddings = {}
        npz_path = target / "state.npz"
        if npz_path.exists():
            npz = np.load(npz_path)
            self.embeddings = npz["embeddings"]
            if "user_profile_embeddings" in npz.files and user_profile_ids:
                profile_embeddings = np.asarray(npz["user_profile_embeddings"], dtype=np.float32)
                self.user_profile_embeddings = {
                    user_id: profile_embeddings[index]
                    for index, user_id in enumerate(user_profile_ids[: len(profile_embeddings)])
                }
            if self.backend != "tfidf":
                try:
                    from sentence_transformers import SentenceTransformer

                    self.embedding_model = SentenceTransformer(self.model_name)
                except Exception:
                    self.embedding_model = None
                    self.backend = "tfidf" if self.vectorizer is not None else "sentence_transformer_numpy"
