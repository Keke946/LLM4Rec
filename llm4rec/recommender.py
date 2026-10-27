from __future__ import annotations

import logging
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm4rec.config import DEFAULT_PATHS, ProjectPaths
from llm4rec.data import build_tag_binarizer, load_videos
from llm4rec.explainer import generate_explanation
from llm4rec.features import FeatureEncoders
from llm4rec.features.encoders import bucketize_duration
from llm4rec.models import CTRMLP, build_ctr_feature_matrix
from llm4rec.ranking import RANKER_REGISTRY
from llm4rec.ranking.base import RankerConfig
from llm4rec.retrieval import RETRIEVER_REGISTRY, reciprocal_rank_fusion
from llm4rec.retrieval.base import BaseRetriever
from llm4rec.rerank import apply_business_reranker
from llm4rec.scoring import compute_final_scores
from llm4rec.text import extract_interest_tags


logger = logging.getLogger(__name__)


class LLM4RecRecommender:
    """Two-stage video recommender with robust local fallbacks."""

    def __init__(
        self,
        paths: ProjectPaths = DEFAULT_PATHS,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        allow_vector_recall: bool = True,
        use_llm_explanations: bool = False,
        dataset: str | None = None,
        processed_dir: str | Path | None = None,
        retriever_dir: str | Path | None = None,
    ) -> None:
        self.paths = paths
        self.embedding_model_name = embedding_model_name
        self.allow_vector_recall = allow_vector_recall
        self.use_llm_explanations = use_llm_explanations
        self.dataset = dataset
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.retriever_dir = Path(retriever_dir) if retriever_dir else None
        self.features_dir = Path(f"data/features/{dataset}") if dataset else None
        self.ranker_root = Path(f"models/{dataset}") if dataset else None
        self.processed_items: pd.DataFrame | None = None
        self.processed_interactions: pd.DataFrame | None = None
        self._retriever_cache: dict[str, BaseRetriever] = {}
        self._ranker_cache: dict[str, Any] = {}

        if self.processed_dir is not None:
            self.processed_items = pd.read_parquet(self.processed_dir / "items.parquet")
            self.processed_interactions = pd.read_parquet(self.processed_dir / "interactions.parquet")
            self.videos = pd.DataFrame()
            self.video_id_map = []
            self.tag_binarizer = None
            self.tag_vocab = []
            self.ctr_model = None
        else:
            self.videos = load_videos(paths.video_csv)
            self.video_id_map = self._load_video_id_map()
            self.tag_binarizer = build_tag_binarizer(self.videos)
            self.tag_vocab = list(self.tag_binarizer.classes_)
            self.ctr_model = self._load_ctr_model()
        self._embedding_model: Any | None = None
        self._faiss_index: Any | None = None

    def recommend(
        self,
        user_interest: str | list[str] | None = None,
        k: int = 5,
        user_id: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
        candidate_pool_k: int | None = None,
        use_llm_explainer: bool = False,
        retrievers: list[str] | None = None,
        ranker: str = "deepfm",
        debug: bool = False,
        filter_seen_splits: list[str] | None = None,
        rerank_mode: str = "retrieval_ranker_fusion",
        retrieval_weight: float = 0.7,
        ranker_weight: float = 0.3,
        reranker: str = "none",
        source_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if self.processed_dir is not None:
            return self._recommend_processed(
                user_id=user_id or "",
                query=query or (user_interest if isinstance(user_interest, str) else None),
                top_k=top_k or k,
                candidate_pool_k=candidate_pool_k,
                retrievers=retrievers or ["popular", "itemcf", "content_faiss"],
                use_llm_explainer=use_llm_explainer,
                ranker=ranker,
                debug=debug,
                filter_seen_splits=filter_seen_splits or ["train", "val"],
                rerank_mode=rerank_mode,
                retrieval_weight=retrieval_weight,
                ranker_weight=ranker_weight,
                reranker=reranker,
                source_weights=source_weights,
            )

        if user_interest is None:
            user_interest = query or ""
        tags = extract_interest_tags(user_interest, self.tag_vocab)
        if not tags:
            return []

        candidate_count = min(len(self.videos), max(k * 4, k))
        candidates = self._recall_candidates(tags, candidate_count)
        if candidates.empty:
            return []

        scores = self._rank_candidates(tags, candidates)
        ranked = candidates.copy()
        ranked["score"] = scores
        ranked = ranked.sort_values(["score", "video_id"], ascending=[False, True]).head(k)

        results: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            score = float(row["score"])
            results.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "score": round(score, 4),
                    "tags": list(row["tags"]),
                    "labels": list(row["labels"]),
                    "reason": generate_explanation(
                        user_tags=tags,
                        video_tags=row["tags"],
                        score=score,
                        title=row["title"],
                        use_llm_fallback=self.use_llm_explanations,
                        allowed_item_ids=[row["video_id"]],
                    ),
                }
            )
        return results

    def _recommend_processed(
        self,
        user_id: str,
        query: str | None,
        top_k: int,
        candidate_pool_k: int | None,
        retrievers: list[str],
        use_llm_explainer: bool,
        ranker: str,
        debug: bool,
        filter_seen_splits: list[str],
        rerank_mode: str,
        retrieval_weight: float,
        ranker_weight: float,
        reranker: str,
        source_weights: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        if self.processed_items is None or self.processed_interactions is None:
            return []
        candidate_pool_k = max(candidate_pool_k or max(200, top_k * 20), top_k)
        seen = self._processed_seen_items(user_id, filter_seen_splits)
        candidate_lists = [
            self._filter_candidates(
                self._load_processed_retriever(name).retrieve(user_id, query=query, top_k=candidate_pool_k),
                seen,
            )
            for name in retrievers
        ]
        merged = reciprocal_rank_fusion(candidate_lists, top_k=candidate_pool_k, weights=source_weights)
        merged = self._filter_candidates(merged, seen)
        candidate_pool_size = len(merged)
        ranker_bundle = self._load_processed_ranker(ranker)
        if ranker_bundle is not None and rerank_mode != "retrieval_only":
            merged = self._rerank_processed_candidates(
                user_id,
                merged,
                ranker_bundle,
                candidate_pool_k,
                rerank_mode,
                retrieval_weight,
                ranker_weight,
                [split for split in filter_seen_splits if split in {"train", "val"}],
            )
        if reranker not in {"", "none", None}:
            item_lookup_for_rerank = self.processed_items.set_index("item_id").to_dict(orient="index")
            item_popularity = Counter(self._processed_item_popularity())
            counts = sorted(item_popularity.values())
            threshold = counts[int(len(counts) * 0.2)] if counts else 0
            tail_items = {item for item, count in item_popularity.items() if count <= threshold}
            ordered_ids = apply_business_reranker(
                [candidate.item_id for candidate in merged],
                item_lookup_for_rerank,
                item_popularity,
                tail_items,
                mode=reranker,
                top_k=len(merged),
            )
            order = {item_id: idx for idx, item_id in enumerate(ordered_ids)}
            merged = sorted(merged, key=lambda candidate: order.get(candidate.item_id, len(order)))
        final_candidates = merged[:top_k]
        item_lookup = self.processed_items.set_index("item_id").to_dict(orient="index")
        recommendations: list[dict[str, Any]] = []
        for rank, candidate in enumerate(final_candidates, start=1):
            meta = item_lookup.get(candidate.item_id, {})
            tags = list(meta.get("tags", [])) if isinstance(meta.get("tags", []), list) else []
            categories = (
                list(meta.get("categories", [])) if isinstance(meta.get("categories", []), list) else []
            )
            title = str(meta.get("title", candidate.item_id))
            explanation = generate_explanation(
                user_tags=extract_interest_tags(query or "", categories + tags),
                video_tags=tags or categories,
                score=candidate.score,
                title=title,
                use_llm_fallback=use_llm_explainer,
                allowed_item_ids=[candidate.item_id],
            )
            row = {
                "item_id": candidate.item_id,
                "title": title,
                "score": round(float(candidate.score), 6),
                "rank": rank,
                "retrieval_sources": candidate.source.split("+"),
                "explanation": explanation,
            }
            if debug:
                row["debug"] = {
                    "ranker": ranker,
                    "candidate_pool_size": candidate_pool_size,
                    "retrieval_sources": candidate.source.split("+"),
                    "retrieval_score": candidate.retrieval_score,
                    "ranker_score": candidate.ranker_score,
                    "final_score": candidate.score,
                    "rerank_mode": rerank_mode,
                    "business_reranker": reranker,
                    "source_weights": source_weights or {},
                }
            recommendations.append(row)
        return recommendations

    def _load_processed_retriever(self, name: str) -> BaseRetriever:
        if name in self._retriever_cache:
            return self._retriever_cache[name]
        if self.retriever_dir is not None:
            for path in [self.retriever_dir / name, self.retriever_dir / f"{name}.pkl"]:
                if path.exists():
                    try:
                        retriever = BaseRetriever.load(path)
                        self._retriever_cache[name] = retriever
                        return retriever
                    except Exception as exc:
                        logger.warning("Failed to load retriever %s from %s: %s. Refitting.", name, path, exc)
                        break
        train = self.processed_interactions[self.processed_interactions["split"] == "train"].reset_index(drop=True)
        retriever = RETRIEVER_REGISTRY[name]().fit(train, self.processed_items)
        self._retriever_cache[name] = retriever
        return retriever

    def _load_processed_ranker(self, name: str):
        if not name or self.features_dir is None or self.ranker_root is None:
            return None
        if name in self._ranker_cache:
            return self._ranker_cache[name]
        model_dir = self.ranker_root / name
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        encoders_path = self.features_dir / "encoders.pkl"
        if not model_path.exists() or not config_path.exists() or not encoders_path.exists():
            return None
        config_raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = RankerConfig(
            num_users=int(config_raw["num_users"]),
            num_items=int(config_raw["num_items"]),
            num_authors=int(config_raw["num_authors"]),
            num_categories=int(config_raw["num_categories"]),
            num_tags=int(config_raw["num_tags"]),
            num_retrieval_sources=int(config_raw.get("num_retrieval_sources", 9)),
            num_duration_buckets=int(config_raw["num_duration_buckets"]),
            dense_feature_dim=int(config_raw.get("dense_feature_dim", 6)),
            embedding_dim=int(config_raw["embedding_dim"]),
            max_history_len=int(config_raw.get("max_history_len", 20)),
            dense_transform=config_raw.get("dense_transform", "none"),
        )
        model = RANKER_REGISTRY[name](config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        bundle = {"model": model, "config": config, "encoders": FeatureEncoders.load(encoders_path)}
        self._ranker_cache[name] = bundle
        return bundle

    def _rerank_processed_candidates(
        self,
        user_id: str,
        candidates,
        ranker_bundle,
        top_k: int,
        rerank_mode: str,
        retrieval_weight: float,
        ranker_weight: float,
        history_splits: list[str],
    ):
        model = ranker_bundle["model"]
        config = ranker_bundle["config"]
        encoders = ranker_bundle["encoders"]
        item_lookup = self.processed_items.set_index("item_id").to_dict(orient="index")
        history = self._processed_history_indices(user_id, encoders, history_splits)
        item_popularity = self._processed_item_popularity()
        user_activity = self._processed_user_activity()
        if not candidates:
            return []
        padded_history = _pad_list(history[-config.max_history_len:], config.max_history_len)
        batch_rows = {
            "user_idx": [],
            "item_idx": [],
            "author_idx": [],
            "duration_bucket": [],
            "category_indices": [],
            "tag_indices": [],
            "history_item_indices": [],
        }
        with torch.no_grad():
            for candidate in candidates:
                meta = item_lookup.get(candidate.item_id, {})
                sources = candidate.source.split("+") if candidate.source else ["unknown"]
                retrieval_score = float(candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score)
                batch_rows["user_idx"].append(encoders.user.transform_one(user_id))
                batch_rows["item_idx"].append(encoders.item.transform_one(candidate.item_id))
                batch_rows["author_idx"].append(encoders.author.transform_one(meta.get("author_id", "")))
                batch_rows["duration_bucket"].append(bucketize_duration(meta.get("duration"), encoders.duration_boundaries))
                batch_rows["category_indices"].append(
                    _pad_list(encoders.category.transform_list(meta.get("categories", []), 8), 8)
                )
                batch_rows["tag_indices"].append(
                    _pad_list(encoders.tag.transform_list(meta.get("tags", []), 16), 16)
                )
                batch_rows.setdefault("retrieval_source_indices", []).append(
                    _pad_list(encoders.retrieval_source.transform_list(sources, 8), 8)
                )
                batch_rows["history_item_indices"].append(padded_history)
                batch_rows.setdefault("dense_features", []).append(
                    [
                        retrieval_score,
                        float(candidate.rank),
                        float(len(sources)),
                        float(item_popularity.get(candidate.item_id, 0)),
                        float(user_activity.get(str(user_id), 0)),
                        float(len(history)),
                    ]
                )
            batch = {
                key: torch.tensor(value, dtype=torch.float32 if key == "dense_features" else torch.long)
                for key, value in batch_rows.items()
            }
            logits = model(batch).detach().cpu().numpy().tolist()
            retrieval_scores = [
                float(candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score)
                for candidate in candidates
            ]
            final_scores = compute_final_scores(
                retrieval_scores,
                [float(logit) for logit in logits],
                mode=rerank_mode,
                retrieval_weight=retrieval_weight,
                ranker_weight=ranker_weight,
            )
            for candidate, logit, final_score in zip(
                candidates,
                logits,
                final_scores,
            ):
                logit = float(logit)
                if candidate.retrieval_score is None:
                    candidate.retrieval_score = candidate.score
                candidate.ranker_score = logit
                candidate.score = float(final_score)
        ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]
        for idx, candidate in enumerate(ranked, start=1):
            candidate.rank = idx
        return ranked

    def _processed_history_indices(
        self,
        user_id: str,
        encoders: FeatureEncoders,
        history_splits: list[str] | None = None,
    ) -> list[int]:
        splits = history_splits or ["train", "val"]
        history = self.processed_interactions[
            (self.processed_interactions["split"].isin(splits))
            & (self.processed_interactions["user_id"].astype(str) == str(user_id))
            & (self.processed_interactions["label"] == 1)
        ].sort_values(["timestamp", "item_id"])
        return [encoders.item.transform_one(item_id) for item_id in history["item_id"].astype(str)]

    def _processed_seen_items(self, user_id: str, splits: list[str]) -> set[str]:
        seen = self.processed_interactions[
            (self.processed_interactions["split"].isin(splits))
            & (self.processed_interactions["user_id"].astype(str) == str(user_id))
            & (self.processed_interactions["label"] == 1)
        ]["item_id"].astype(str)
        return set(seen)

    def _processed_item_popularity(self) -> dict[str, int]:
        train = self.processed_interactions[
            (self.processed_interactions["split"] == "train") & (self.processed_interactions["label"] == 1)
        ]
        return train.groupby("item_id").size().to_dict()

    def _processed_user_activity(self) -> dict[str, int]:
        train = self.processed_interactions[
            (self.processed_interactions["split"] == "train") & (self.processed_interactions["label"] == 1)
        ]
        return train.groupby("user_id").size().to_dict()

    def _filter_candidates(self, candidates, seen: set[str]):
        if not seen:
            return candidates
        filtered = [candidate for candidate in candidates if candidate.item_id not in seen]
        for rank, candidate in enumerate(filtered, start=1):
            candidate.rank = rank
        return filtered

    def _load_video_id_map(self) -> list[str]:
        if self.paths.video_id_map.exists():
            return [
                line.strip()
                for line in self.paths.video_id_map.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return self.videos["video_id"].tolist()

    def _load_ctr_model(self) -> CTRMLP | None:
        model_path = Path(self.paths.ctr_model)
        if not model_path.exists():
            logger.info("CTR model not found. Falling back to heuristic ranking.")
            return None

        input_dim = len(self.tag_vocab) * 2
        model = CTRMLP(input_dim)
        try:
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as exc:
            logger.warning("Could not load CTR model from %s: %s", model_path, exc)
            return None

    def _load_vector_backend(self) -> tuple[Any, Any] | None:
        if not self.allow_vector_recall:
            return None
        if self._faiss_index is not None and self._embedding_model is not None:
            return self._faiss_index, self._embedding_model
        if not self.paths.faiss_index.exists():
            return None

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.info("Vector recall dependencies are unavailable: %s", exc)
            return None

        self._faiss_index = faiss.read_index(str(self.paths.faiss_index))
        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._faiss_index, self._embedding_model

    def _recall_candidates(self, tags: list[str], count: int) -> pd.DataFrame:
        backend = self._load_vector_backend()
        if backend is not None:
            index, embedding_model = backend
            query_vec = embedding_model.encode([" ".join(tags)], normalize_embeddings=True)
            query_vec = np.asarray(query_vec, dtype=np.float32)
            _, indices = index.search(query_vec, count)
            video_ids = [
                self.video_id_map[idx]
                for idx in indices[0]
                if 0 <= idx < len(self.video_id_map)
            ]
            if video_ids:
                order = {video_id: pos for pos, video_id in enumerate(video_ids)}
                candidates = self.videos[self.videos["video_id"].isin(video_ids)].copy()
                candidates["_recall_order"] = candidates["video_id"].map(order)
                return candidates.sort_values("_recall_order").drop(columns=["_recall_order"])

        return self._heuristic_recall(tags, count)

    def _heuristic_recall(self, tags: list[str], count: int) -> pd.DataFrame:
        tag_set = set(tags)
        scored = self.videos.copy()

        def score_row(row: pd.Series) -> float:
            video_tags = {str(tag).lower() for tag in row["tags"]}
            text = f"{row['title']} {row['description']}".lower()
            tag_overlap = len(tag_set & video_tags) / max(1, len(tag_set))
            text_overlap = sum(1 for tag in tag_set if tag in text) / max(1, len(tag_set))
            return 0.75 * tag_overlap + 0.25 * text_overlap

        scored["_recall_score"] = scored.apply(score_row, axis=1)
        return scored.sort_values(["_recall_score", "video_id"], ascending=[False, True]).head(count)

    def _rank_candidates(self, tags: list[str], candidates: pd.DataFrame) -> np.ndarray:
        if self.ctr_model is not None:
            features = build_ctr_feature_matrix(tags, candidates["tags"].tolist(), self.tag_binarizer)
            tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                return self.ctr_model(tensor).detach().cpu().numpy().reshape(-1)

        tag_set = set(tags)
        scores: list[float] = []
        for _, row in candidates.iterrows():
            video_tags = {str(tag).lower() for tag in row["tags"]}
            title_desc = f"{row['title']} {row['description']}".lower()
            tag_score = len(tag_set & video_tags) / max(1, len(tag_set))
            text_score = sum(1 for tag in tag_set if tag in title_desc) / max(1, len(tag_set))
            popularity_prior = 0.05 if "compilation" in video_tags or "tutorial" in video_tags else 0.0
            scores.append(min(1.0, 0.8 * tag_score + 0.15 * text_score + popularity_prior))
        return np.asarray(scores, dtype=np.float32)


def _pad_list(values: list[int], length: int) -> list[int]:
    values = values[:length]
    return values + [0] * (length - len(values))
