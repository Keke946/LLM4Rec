from __future__ import annotations

import json
import logging
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

from llm4rec.evaluation.metrics import (
    average_popularity,
    catalog_coverage,
    category_diversity,
    item_coverage_count,
    novelty,
    ranking_metrics,
    tail_ratio,
)
from llm4rec.features import FeatureEncoders
from llm4rec.features.encoders import bucketize_duration
from llm4rec.ranking import RANKER_REGISTRY
from llm4rec.ranking.base import RankerConfig
from llm4rec.ranking.gbdt_features import gbdt_feature_row
from llm4rec.ranking.lambdarank import load_lambdarank_model, make_lambdarank_features
from llm4rec.retrieval import RETRIEVER_REGISTRY, reciprocal_rank_fusion
from llm4rec.retrieval.base import BaseRetriever
from llm4rec.rerank import apply_business_reranker
from llm4rec.scoring import compute_final_scores

logger = logging.getLogger(__name__)


class OfflineEvaluator:
    def __init__(
        self,
        processed_dir: str | Path,
        retriever_dir: str | Path | None = None,
        features_dir: str | Path | None = None,
        ranker_root: str | Path | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.retriever_dir = Path(retriever_dir) if retriever_dir else None
        self.features_dir = Path(features_dir) if features_dir else None
        self.ranker_root = Path(ranker_root) if ranker_root else None
        self.items = pd.read_parquet(self.processed_dir / "items.parquet")
        self.interactions = pd.read_parquet(self.processed_dir / "interactions.parquet")
        self.train = self.interactions[self.interactions["split"] == "train"].reset_index(drop=True)
        self.val = self.interactions[self.interactions["split"] == "val"].reset_index(drop=True)
        self.test = self.interactions[self.interactions["split"] == "test"].reset_index(drop=True)
        self.item_meta = self.items.set_index("item_id").to_dict(orient="index")
        self._retriever_cache: dict[str, BaseRetriever] = {}
        self.popularity = Counter(self.train[self.train["label"] == 1]["item_id"].astype(str))
        self.user_activity = Counter(self.train[self.train["label"] == 1]["user_id"].astype(str))
        counts = sorted(self.popularity.values())
        threshold = counts[int(len(counts) * 0.2)] if counts else 0
        self.tail_items = {item for item, count in self.popularity.items() if count <= threshold}

    def evaluate_retrievers(
        self,
        retriever_names: list[str],
        top_k: int | None = None,
        query_by_user: dict[str, str] | None = None,
        ranker_name: str | None = None,
        candidate_pool_k: int = 500,
        cutoffs: list[int] | None = None,
        final_top_k: int = 10,
        filter_seen_splits: list[str] | None = None,
        retriever_top_k: int | None = None,
        rerank_mode: str = "retrieval_ranker_fusion",
        retrieval_weight: float = 0.7,
        ranker_weight: float = 0.3,
        reranker: str = "none",
        source_weights: dict[str, float] | None = None,
    ) -> dict[str, object]:
        if top_k is not None:
            candidate_pool_k = max(candidate_pool_k, top_k)
            final_top_k = top_k
        cutoffs = cutoffs or [5, 10, 20, 50, 100]
        max_cutoff = max(cutoffs)
        candidate_pool_k = max(candidate_pool_k, max_cutoff, final_top_k)
        retriever_top_k = retriever_top_k or candidate_pool_k
        filter_seen_splits = filter_seen_splits or ["train", "val"]
        query_by_user = query_by_user or {}
        retrievers = [self._load_or_fit_retriever(name) for name in retriever_names]
        excluded_by_user = self._positive_items_by_user(filter_seen_splits)
        relevant_by_user = {
            str(user_id): set(group[group["label"] == 1]["item_id"].astype(str))
            for user_id, group in self.test.groupby("user_id")
        }
        recommended_by_user: dict[str, list[str]] = {}
        ranked_by_user: dict[str, list[str]] = {}
        final_by_user: dict[str, list[str]] = {}
        retrieved_candidates_by_user: dict[str, list[str]] = {}
        candidate_count_by_user: dict[str, int] = {}
        latencies: list[float] = []
        candidate_counts: list[int] = []
        ranker_bundle = self._load_ranker(ranker_name) if ranker_name else None

        if ranker_name and ranker_bundle is None:
            raise FileNotFoundError(f"Requested ranker {ranker_name!r} has no loadable artifact.")

        for user_id, relevant in relevant_by_user.items():
            if not relevant:
                continue
            start = time.perf_counter()
            seen_for_user = excluded_by_user.get(user_id, set())
            expanded_retriever_top_k = retriever_top_k + len(seen_for_user)
            candidate_lists = [
                self._filter_candidate_objects(
                    retriever.retrieve(user_id, query=query_by_user.get(user_id), top_k=expanded_retriever_top_k),
                    seen_for_user,
                )
                for retriever in retrievers
            ]
            merged = reciprocal_rank_fusion(candidate_lists, top_k=candidate_pool_k, weights=source_weights)
            merged = self._filter_candidate_objects(merged, seen_for_user)
            candidate_counts.append(len(merged))
            candidate_count_by_user[user_id] = len(merged)
            retrieved_candidates_by_user[user_id] = [candidate.item_id for candidate in merged]
            retrieval_items = [candidate.item_id for candidate in merged[:max_cutoff]]
            recommended_by_user[user_id] = retrieval_items
            base_items = retrieval_items
            if ranker_bundle is not None and rerank_mode != "retrieval_only":
                ranked_by_user[user_id] = self._rerank_with_ranker(
                    user_id,
                    merged[:candidate_pool_k],
                    ranker_bundle,
                    max_cutoff,
                    rerank_mode,
                    retrieval_weight,
                    ranker_weight,
                    [split for split in filter_seen_splits if split in {"train", "val"}],
                )
                base_items = ranked_by_user[user_id]
            final_by_user[user_id] = apply_business_reranker(
                base_items,
                self.item_meta,
                self.popularity,
                self.tail_items,
                mode=reranker,
                top_k=max_cutoff,
            )
            latencies.append((time.perf_counter() - start) * 1000)

        per_user = [
            ranking_metrics(recs, relevant_by_user[user_id], cutoffs)
            for user_id, recs in recommended_by_user.items()
        ]
        summary = _mean_metrics(per_user)
        summary.update(
            {
                "CatalogCoverage": catalog_coverage(recommended_by_user, set(self.items["item_id"].astype(str))),
                "ItemCoverage": item_coverage_count(recommended_by_user),
                "AveragePopularity": average_popularity(recommended_by_user, self.popularity),
                "TailRatio": tail_ratio(recommended_by_user, self.tail_items),
                "Diversity": category_diversity(recommended_by_user, self.item_meta),
                "Novelty": novelty(recommended_by_user, self.popularity),
                "avg_candidates": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                "latency_p50_ms": _percentile(latencies, 50),
                "latency_p95_ms": _percentile(latencies, 95),
            }
        )
        ranked_summary = None
        if ranked_by_user:
            ranked_summary = _mean_metrics(
                [
                    ranking_metrics(recs, relevant_by_user[user_id], cutoffs)
                    for user_id, recs in ranked_by_user.items()
                ]
            )
            ranked_summary.update(
                {
                    "CatalogCoverage": catalog_coverage(ranked_by_user, set(self.items["item_id"].astype(str))),
                    "ItemCoverage": item_coverage_count(ranked_by_user),
                    "AveragePopularity": average_popularity(ranked_by_user, self.popularity),
                    "TailRatio": tail_ratio(ranked_by_user, self.tail_items),
                    "Diversity": category_diversity(ranked_by_user, self.item_meta),
                    "Novelty": novelty(ranked_by_user, self.popularity),
                    "avg_candidates": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                    "latency_p50_ms": _percentile(latencies, 50),
                    "latency_p95_ms": _percentile(latencies, 95),
                }
            )
        final_summary = None
        if reranker not in {"", "none", None} and final_by_user:
            final_summary = _mean_metrics(
                [
                    ranking_metrics(recs, relevant_by_user[user_id], cutoffs)
                    for user_id, recs in final_by_user.items()
                ]
            )
            final_summary.update(
                {
                    "CatalogCoverage": catalog_coverage(final_by_user, set(self.items["item_id"].astype(str))),
                    "ItemCoverage": item_coverage_count(final_by_user),
                    "AveragePopularity": average_popularity(final_by_user, self.popularity),
                    "TailRatio": tail_ratio(final_by_user, self.tail_items),
                    "Diversity": category_diversity(final_by_user, self.item_meta),
                    "Novelty": novelty(final_by_user, self.popularity),
                    "avg_candidates": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                    "latency_p50_ms": _percentile(latencies, 50),
                    "latency_p95_ms": _percentile(latencies, 95),
                }
            )
        effective_summary = final_summary or ranked_summary or summary
        before_business = ranked_summary or summary
        return {
            "retrievers": retriever_names,
            "ranker": ranker_name,
            "candidate_pool_k": candidate_pool_k,
            "final_top_k": final_top_k,
            "cutoffs": cutoffs,
            "filter_seen_splits": filter_seen_splits,
            "rerank_mode": rerank_mode,
            "retrieval_weight": retrieval_weight,
            "ranker_weight": ranker_weight,
            "source_weights": source_weights or {},
            "reranker": reranker,
            "ranker_available": bool(ranker_bundle) if ranker_name else False,
            "summary": effective_summary,
            "retrieval_only_summary": summary,
            "retrieval_plus_rank_summary": ranked_summary,
            "business_rerank_summary": final_summary,
            "business_rerank_before_summary": before_business,
            "failure_cases": self._failure_cases(
                final_by_user or ranked_by_user or recommended_by_user,
                relevant_by_user,
                retrieved_candidates_by_user,
                candidate_count_by_user,
            ),
        }

    def _load_or_fit_retriever(self, name: str) -> BaseRetriever:
        if name in self._retriever_cache:
            return self._retriever_cache[name]
        if self.retriever_dir:
            for path in [self.retriever_dir / name, self.retriever_dir / f"{name}.pkl"]:
                if path.exists():
                    try:
                        retriever = BaseRetriever.load(path)
                        self._retriever_cache[name] = retriever
                        return retriever
                    except Exception as exc:
                        logger.warning("Failed to load retriever %s from %s: %s. Refitting.", name, path, exc)
                        break
        retriever = RETRIEVER_REGISTRY[name]().fit(self.train, self.items)
        self._retriever_cache[name] = retriever
        return retriever

    def _positive_items_by_user(self, splits: list[str]) -> dict[str, set[str]]:
        selected = self.interactions[
            (self.interactions["split"].isin(splits)) & (self.interactions["label"] == 1)
        ]
        return {
            str(user_id): set(group["item_id"].astype(str))
            for user_id, group in selected.groupby("user_id")
        }

    def _filter_candidate_objects(self, candidates, seen: set[str]):
        if not seen:
            return candidates
        filtered = [candidate for candidate in candidates if candidate.item_id not in seen]
        for rank, candidate in enumerate(filtered, start=1):
            candidate.rank = rank
        return filtered

    def _failure_cases(
        self,
        recommended_by_user: dict[str, list[str]],
        relevant_by_user: dict[str, set[str]],
        retrieved_candidates_by_user: dict[str, list[str]] | None = None,
        candidate_count_by_user: dict[str, int] | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        retrieved_candidates_by_user = retrieved_candidates_by_user or {}
        candidate_count_by_user = candidate_count_by_user or {}
        cases = []
        for user_id, relevant in relevant_by_user.items():
            recs = recommended_by_user.get(user_id, [])
            if relevant and not (set(recs) & relevant):
                retrieved = retrieved_candidates_by_user.get(user_id, [])
                positive_retrieved = set(retrieved) & relevant
                reason = self._miss_reason(user_id, relevant, recs[:10], positive_retrieved)
                cases.append(
                    {
                        "user_id": user_id,
                        "ground_truth_items": sorted(relevant)[:10],
                        "relevant_items": sorted(relevant)[:10],
                        "retrieved_candidates_count": int(candidate_count_by_user.get(user_id, len(retrieved))),
                        "positive_retrieved_count": int(len(positive_retrieved)),
                        "top10_items": recs[:10],
                        "recommended_items": recs[:10],
                        "miss_reason": reason,
                    }
                )
            if len(cases) >= limit:
                break
        return cases

    def _miss_reason(
        self,
        user_id: str,
        relevant: set[str],
        top10: list[str],
        positive_retrieved: set[str],
    ) -> str:
        if not positive_retrieved:
            return "not_retrieved"
        history_len = self._history_len(user_id, ["train", "val"])
        if history_len < 2:
            return "insufficient_user_history"
        relevant_popularity = [int(self.popularity.get(item_id, 0)) for item_id in relevant]
        if relevant_popularity and min(relevant_popularity) == 0:
            return "cold_item"
        if relevant and all(item_id in self.tail_items for item_id in relevant):
            return "tail_item"
        top_popularity = [int(self.popularity.get(item_id, 0)) for item_id in top10]
        if top_popularity and relevant_popularity:
            popularity_values = list(self.popularity.values())
            high = pd.Series(popularity_values).quantile(0.8) if popularity_values else 0
            low = pd.Series(popularity_values).quantile(0.2) if popularity_values else 0
            if sum(top_popularity) / len(top_popularity) >= high and sum(relevant_popularity) / len(relevant_popularity) <= low:
                return "popularity_bias"
        if self._category_mismatch(user_id, relevant, top10):
            return "category_mismatch"
        return "retrieved_but_low_rank"

    def _category_mismatch(self, user_id: str, relevant: set[str], top10: list[str]) -> bool:
        history_rows = self.interactions[
            (self.interactions["split"].isin(["train", "val"]))
            & (self.interactions["user_id"].astype(str) == str(user_id))
            & (self.interactions["label"] == 1)
        ]
        history_categories: set[str] = set()
        for item_id in history_rows["item_id"].astype(str).tolist()[-50:]:
            history_categories.update(_string_values(self.item_meta.get(item_id, {}).get("categories", [])))
        relevant_categories = {
            category
            for item_id in relevant
            for category in _string_values(self.item_meta.get(item_id, {}).get("categories", []))
        }
        top_categories = {
            category
            for item_id in top10
            for category in _string_values(self.item_meta.get(item_id, {}).get("categories", []))
        }
        if not relevant_categories or not top_categories:
            return False
        if history_categories and relevant_categories & history_categories:
            return bool(not (top_categories & relevant_categories))
        return bool(not (top_categories & relevant_categories))

    def _load_ranker(self, ranker_name: str | None):
        if not ranker_name or self.features_dir is None or self.ranker_root is None:
            return None
        model_dir = self.ranker_root / ranker_name
        gbdt_model_path = model_dir / "model.pkl"
        gbdt_config_path = model_dir / "config.json"
        encoders_path = self.features_dir / "encoders.pkl"
        if ranker_name == "gbdt" and gbdt_model_path.exists() and gbdt_config_path.exists():
            with gbdt_model_path.open("rb") as file:
                model = pickle.load(file)
            config = json.loads(gbdt_config_path.read_text(encoding="utf-8"))
            encoders = FeatureEncoders.load(encoders_path) if encoders_path.exists() else None
            return {
                "kind": "gbdt",
                "model": model,
                "config": config,
                "encoders": encoders,
                "feature_columns": config.get("feature_columns", []),
                "source_names": config.get("source_names", []),
            }
        lambdarank_model_path = model_dir / "model.pkl"
        lambdarank_schema_path = model_dir / "feature_schema.json"
        lambdarank_config_path = model_dir / "config.json"
        if lambdarank_model_path.exists() and lambdarank_schema_path.exists():
            config = (
                json.loads(lambdarank_config_path.read_text(encoding="utf-8"))
                if lambdarank_config_path.exists()
                else {}
            )
            if ranker_name == "lambdarank" or config.get("model") == "lambdarank":
                model, schema = load_lambdarank_model(model_dir)
                encoders = FeatureEncoders.load(encoders_path) if encoders_path.exists() else None
                return {
                    "kind": "lambdarank",
                    "model": model,
                    "schema": schema,
                    "config": config,
                    "encoders": encoders,
                    "feature_columns": schema.get("feature_columns", []),
                    "source_names": schema.get("source_names", []),
                }
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
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
        model = RANKER_REGISTRY[ranker_name](config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        encoders = FeatureEncoders.load(encoders_path)
        return {"kind": "torch", "model": model, "config": config, "encoders": encoders}

    def _rerank_with_ranker(
        self,
        user_id: str,
        candidates,
        ranker_bundle,
        output_k: int,
        rerank_mode: str,
        retrieval_weight: float,
        ranker_weight: float,
        history_splits: list[str],
    ) -> list[str]:
        if ranker_bundle.get("kind") == "gbdt":
            return self._rerank_with_gbdt(
                user_id,
                candidates,
                ranker_bundle,
                output_k,
                rerank_mode,
                retrieval_weight,
                ranker_weight,
                history_splits,
            )
        if ranker_bundle.get("kind") == "lambdarank":
            return self._rerank_with_lambdarank(
                user_id,
                candidates,
                ranker_bundle,
                output_k,
                rerank_mode,
                retrieval_weight,
                ranker_weight,
                history_splits,
            )
        model = ranker_bundle["model"]
        config = ranker_bundle["config"]
        encoders = ranker_bundle["encoders"]
        if not candidates:
            return []
        history = _pad(self._history_indices(user_id, encoders, history_splits)[-config.max_history_len:], config.max_history_len)
        batch_rows = {
            "user_idx": [],
            "item_idx": [],
            "author_idx": [],
            "duration_bucket": [],
            "category_indices": [],
            "tag_indices": [],
            "history_item_indices": [],
        }
        item_ids = []
        for candidate in candidates:
            meta = self.item_meta.get(candidate.item_id, {})
            item_ids.append(candidate.item_id)
            sources = candidate.source.split("+") if candidate.source else ["unknown"]
            retrieval_score = float(candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score)
            batch_rows["user_idx"].append(encoders.user.transform_one(user_id))
            batch_rows["item_idx"].append(encoders.item.transform_one(candidate.item_id))
            batch_rows["author_idx"].append(encoders.author.transform_one(meta.get("author_id", "")))
            batch_rows["duration_bucket"].append(bucketize_duration(meta.get("duration"), encoders.duration_boundaries))
            batch_rows["category_indices"].append(_pad(encoders.category.transform_list(meta.get("categories", []), 8), 8))
            batch_rows["tag_indices"].append(_pad(encoders.tag.transform_list(meta.get("tags", []), 16), 16))
            batch_rows.setdefault("retrieval_source_indices", []).append(
                _pad(encoders.retrieval_source.transform_list(sources, 8), 8)
            )
            batch_rows["history_item_indices"].append(history)
            batch_rows.setdefault("dense_features", []).append(
                [
                    retrieval_score,
                    float(candidate.rank),
                    float(len(sources)),
                    float(self.popularity.get(candidate.item_id, 0)),
                    float(self.user_activity.get(str(user_id), 0)),
                    float(len([item for item in history if item])),
                ]
            )
        batch = {
            key: torch.tensor(value, dtype=torch.float32 if key == "dense_features" else torch.long)
            for key, value in batch_rows.items()
        }
        with torch.no_grad():
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
        head_k = _head_rerank_k(rerank_mode)
        if head_k is not None:
            return _ranker_head_order(item_ids, [float(logit) for logit in logits], output_k, head_k)
        scored = []
        for item_id, final_score in zip(item_ids, final_scores):
            scored.append((item_id, final_score))
        return [item_id for item_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:output_k]]

    def _rerank_with_gbdt(
        self,
        user_id: str,
        candidates,
        ranker_bundle,
        output_k: int,
        rerank_mode: str,
        retrieval_weight: float,
        ranker_weight: float,
        history_splits: list[str],
    ) -> list[str]:
        import pandas as pd

        if not candidates:
            return []
        source_names = [str(name) for name in ranker_bundle.get("source_names", [])]
        feature_columns = [str(name) for name in ranker_bundle.get("feature_columns", [])]
        encoders = ranker_bundle.get("encoders")
        history_len = min(50, self._history_len(user_id, history_splits))
        history_profile = self._history_content_profile(user_id, history_splits, encoders)
        rows = []
        item_ids = []
        retrieval_scores = []
        candidate_pool_size = len(candidates)
        for candidate in candidates:
            meta = self.item_meta.get(candidate.item_id, {})
            sources = candidate.source.split("+") if candidate.source else ["unknown"]
            retrieval_score = float(candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score)
            category_indices = (
                encoders.category.transform_list(meta.get("categories", []), 8)
                if encoders
                else meta.get("categories", [])
            )
            tag_indices = (
                encoders.tag.transform_list(meta.get("tags", []), 16)
                if encoders
                else meta.get("tags", [])
            )
            row = {
                "retrieval_score": retrieval_score,
                "retrieval_sources": sources,
                "retrieval_source": "+".join(sources),
                "source_rank": float(candidate.rank),
                "rank": float(candidate.rank),
                "source_count": float(len(sources)),
                "item_popularity": float(self.popularity.get(candidate.item_id, 0)),
                "user_activity_level": float(self.user_activity.get(str(user_id), 0)),
                "history_len": float(history_len),
                "duration_bucket": float(
                    bucketize_duration(meta.get("duration"), encoders.duration_boundaries if encoders else [])
                ),
                "category_indices": category_indices,
                "tag_indices": tag_indices,
                **_content_cross_features(history_profile, category_indices, tag_indices),
                "candidate_pool_size": float(candidate_pool_size),
            }
            rows.append(gbdt_feature_row(row, source_names))
            item_ids.append(candidate.item_id)
            retrieval_scores.append(retrieval_score)
        features = pd.DataFrame(rows)
        for column in feature_columns:
            if column not in features.columns:
                features[column] = 0.0
        probabilities = ranker_bundle["model"].predict_proba(features[feature_columns].fillna(0.0))[:, 1]
        head_k = _head_rerank_k(rerank_mode)
        if head_k is not None:
            return _ranker_head_order(item_ids, [float(probability) for probability in probabilities], output_k, head_k)
        ranker_logits = [_safe_logit(float(probability)) for probability in probabilities]
        final_scores = compute_final_scores(
            retrieval_scores,
            ranker_logits,
            mode=rerank_mode,
            retrieval_weight=retrieval_weight,
            ranker_weight=ranker_weight,
        )
        scored = list(zip(item_ids, final_scores))
        return [item_id for item_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:output_k]]

    def _rerank_with_lambdarank(
        self,
        user_id: str,
        candidates,
        ranker_bundle,
        output_k: int,
        rerank_mode: str,
        retrieval_weight: float,
        ranker_weight: float,
        history_splits: list[str],
    ) -> list[str]:
        import pandas as pd

        if not candidates:
            return []
        encoders = ranker_bundle.get("encoders")
        schema = ranker_bundle["schema"]
        history_len = min(50, self._history_len(user_id, history_splits))
        history_profile = self._history_content_profile(user_id, history_splits, encoders)
        rows = []
        item_ids = []
        retrieval_scores = []
        candidate_pool_size = len(candidates)
        for candidate in candidates:
            meta = self.item_meta.get(candidate.item_id, {})
            sources = candidate.source.split("+") if candidate.source else ["unknown"]
            retrieval_score = float(candidate.retrieval_score if candidate.retrieval_score is not None else candidate.score)
            category_indices = (
                encoders.category.transform_list(meta.get("categories", []), 8)
                if encoders
                else meta.get("categories", [])
            )
            tag_indices = (
                encoders.tag.transform_list(meta.get("tags", []), 16)
                if encoders
                else meta.get("tags", [])
            )
            author_idx = (
                encoders.author.transform_one(meta.get("author_id", ""))
                if encoders
                else 0
            )
            row = {
                "user_id": user_id,
                "item_id": candidate.item_id,
                "retrieval_score": retrieval_score,
                "retrieval_sources": sources,
                "retrieval_source": "+".join(sources),
                "source_rank": float(candidate.rank),
                "rank": float(candidate.rank),
                "source_count": float(len(sources)),
                "item_popularity": float(self.popularity.get(candidate.item_id, 0)),
                "user_activity_level": float(self.user_activity.get(str(user_id), 0)),
                "history_len": float(history_len),
                "duration_bucket": float(
                    bucketize_duration(meta.get("duration"), encoders.duration_boundaries if encoders else [])
                ),
                "author_idx": float(author_idx),
                "category_indices": category_indices,
                "tag_indices": tag_indices,
                **_content_cross_features(history_profile, category_indices, tag_indices),
                "candidate_pool_size": float(candidate_pool_size),
            }
            rows.append(row)
            item_ids.append(candidate.item_id)
            retrieval_scores.append(retrieval_score)
        features = make_lambdarank_features(pd.DataFrame(rows), schema)
        scores = [float(value) for value in ranker_bundle["model"].predict(features)]
        head_k = _head_rerank_k(rerank_mode)
        if head_k is not None:
            return _ranker_head_order(item_ids, scores, output_k, head_k)
        final_scores = compute_final_scores(
            retrieval_scores,
            scores,
            mode=rerank_mode,
            retrieval_weight=retrieval_weight,
            ranker_weight=ranker_weight,
        )
        scored = list(zip(item_ids, final_scores))
        return [item_id for item_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:output_k]]

    def _history_indices(
        self,
        user_id: str,
        encoders: FeatureEncoders,
        history_splits: list[str] | None = None,
    ) -> list[int]:
        splits = history_splits or ["train", "val"]
        history = self.interactions[
            (self.interactions["split"].isin(splits))
            & (self.interactions["user_id"].astype(str) == str(user_id))
            & (self.interactions["label"] == 1)
        ].sort_values(["timestamp", "item_id"])["item_id"].astype(str)
        return [encoders.item.transform_one(item_id) for item_id in history]

    def _history_len(self, user_id: str, history_splits: list[str] | None = None) -> int:
        splits = history_splits or ["train", "val"]
        return int(
            len(
                self.interactions[
                    (self.interactions["split"].isin(splits))
                    & (self.interactions["user_id"].astype(str) == str(user_id))
                    & (self.interactions["label"] == 1)
                ]
            )
        )

    def _history_content_profile(
        self,
        user_id: str,
        history_splits: list[str] | None,
        encoders: FeatureEncoders | None,
    ) -> dict[str, set[int]]:
        splits = history_splits or ["train", "val"]
        rows = self.interactions[
            (self.interactions["split"].isin(splits))
            & (self.interactions["user_id"].astype(str) == str(user_id))
            & (self.interactions["label"] == 1)
        ].sort_values(["timestamp", "item_id"])
        categories: set[int] = set()
        tags: set[int] = set()
        for item_id in rows["item_id"].astype(str).tolist()[-50:]:
            meta = self.item_meta.get(item_id, {})
            if encoders:
                categories.update(_nonzero_ints(encoders.category.transform_list(meta.get("categories", []), 8)))
                tags.update(_nonzero_ints(encoders.tag.transform_list(meta.get("tags", []), 16)))
        return {"categories": categories, "tags": tags}


def _mean_metrics(per_user: list[dict[str, float]]) -> dict[str, float]:
    if not per_user:
        return {}
    keys = sorted({key for row in per_user for key in row})
    return {key: float(sum(row.get(key, 0.0) for row in per_user) / len(per_user)) for key in keys}


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    series = pd.Series(values)
    return float(series.quantile(percentile / 100.0))


def _pad(values: list[int], length: int) -> list[int]:
    values = values[:length]
    return values + [0] * (length - len(values))


def _safe_logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _head_rerank_k(mode: str) -> int | None:
    prefix = "ranker_top_"
    if not str(mode).startswith(prefix):
        return None
    try:
        return max(1, int(str(mode)[len(prefix) :]))
    except ValueError:
        return None


def _ranker_head_order(item_ids: list[str], ranker_scores: list[float], output_k: int, head_k: int) -> list[str]:
    head = list(zip(item_ids[:head_k], ranker_scores[:head_k]))
    reranked_head = [item_id for item_id, _ in sorted(head, key=lambda pair: pair[1], reverse=True)]
    return [*reranked_head, *item_ids[head_k:]][:output_k]


def _content_cross_features(
    history_profile: dict[str, set[int]],
    category_indices,
    tag_indices,
) -> dict[str, float]:
    history_categories = history_profile.get("categories", set())
    history_tags = history_profile.get("tags", set())
    item_categories = set(_nonzero_ints(category_indices))
    item_tags = set(_nonzero_ints(tag_indices))
    return {
        "category_overlap": float(len(history_categories & item_categories)),
        "tag_overlap": float(len(history_tags & item_tags)),
        "category_jaccard": _jaccard(history_categories, item_categories),
        "tag_jaccard": _jaccard(history_tags, item_tags),
        "history_category_count": float(len(history_categories)),
        "history_tag_count": float(len(history_tags)),
    }


def _nonzero_ints(values) -> list[int]:
    if values is None:
        return []
    result: list[int] = []
    for value in values:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric:
            result.append(numeric)
    return result


def _string_values(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values if str(value)]
    if hasattr(values, "tolist"):
        return _string_values(values.tolist())
    return [str(values)] if str(values) else []


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def save_evaluation_outputs(result: dict[str, object], output_dir: str | Path, dataset: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    rows.append({"stage": "retrieval_only", **result["retrieval_only_summary"]})
    if result.get("retrieval_plus_rank_summary"):
        rows.append({"stage": "retrieval_plus_rank", **result["retrieval_plus_rank_summary"]})
    if result.get("business_rerank_summary"):
        rows.append({"stage": f"business_rerank_{result.get('reranker')}", **result["business_rerank_summary"]})
    summary = pd.DataFrame(rows)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "retrievers", ",".join(result["retrievers"]))
    summary.insert(2, "ranker", result.get("ranker") or "")
    summary.insert(3, "candidate_pool_k", result.get("candidate_pool_k", ""))
    summary.insert(4, "final_top_k", result.get("final_top_k", ""))
    summary.insert(5, "rerank_mode", result.get("rerank_mode", ""))
    summary.insert(6, "business_reranker", result.get("reranker", "none"))
    summary.to_csv(output / "metrics_summary.csv", index=False)
    _write_report(result, output / "experiment_report.md", dataset)


def _write_report(result: dict[str, object], path: Path, dataset: str) -> None:
    summary = result["summary"]
    rows = "\n".join(f"| {key} | {value:.6f} |" for key, value in sorted(summary.items()))
    retrieval_rows = "\n".join(
        f"| {key} | {value:.6f} |" for key, value in sorted(result["retrieval_only_summary"].items())
    )
    data_meta = result.get("data_meta", {})
    data_rows = "\n".join(
        f"| {key} | {value} |"
        for key, value in data_meta.items()
        if key in {"num_users", "num_items", "num_interactions", "positive_rate", "split_stats"}
    ) or "| n/a | n/a |"
    model_config = result.get("model_config", {})
    config_rows = "\n".join(f"| {key} | {value} |" for key, value in model_config.items()) or "| n/a | n/a |"
    failures = "\n".join(
        f"- user={case['user_id']} relevant={case['relevant_items']} recommended={case['recommended_items']}"
        for case in result.get("failure_cases", [])
    ) or "- No failure cases found in sampled users."
    text = f"""# LLM4Rec Experiment Report

Dataset: `{dataset}`

Retrievers: `{", ".join(result["retrievers"])}`

Ranker: `{result.get("ranker") or "none"}`

Candidate pool K: `{result.get("candidate_pool_k")}`

Final display Top-K: `{result.get("final_top_k")}`

Metric cutoffs: `{result.get("cutoffs")}`

Filtered seen splits: `{result.get("filter_seen_splits")}`

Rerank mode: `{result.get("rerank_mode")}` (`retrieval_weight={result.get("retrieval_weight")}`, `ranker_weight={result.get("ranker_weight")}`)

Business reranker: `{result.get("reranker", "none")}`

RRF source weights: `{json.dumps(result.get("source_weights", {}), ensure_ascii=False, sort_keys=True)}`

## Data Statistics

| Field | Value |
|---|---:|
{data_rows}

## Model Config

| Field | Value |
|---|---|
{config_rows}

## Primary Metrics

| Metric | Value |
|---|---:|
{rows}

## Retrieval-Only Metrics

| Metric | Value |
|---|---:|
{retrieval_rows}

## Failure Cases

{failures}

## Notes

- Evaluation uses the test split only.
- Training positives are filtered by retrievers before recommendation.
- LLM is not used to generate item IDs.
"""
    path.write_text(text, encoding="utf-8")
