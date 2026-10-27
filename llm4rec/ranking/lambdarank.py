from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm4rec.ranking.gbdt_features import feature_columns, infer_source_names, sources_from_value


SCHEMA_VERSION = 1

OPTIONAL_NUMERIC_FEATURES = [
    "author_idx",
    "author_count",
    "author_overlap",
    "item_duration",
    "item_exposure_count",
    "item_positive_rate",
    "item_avg_watch_ratio",
    "item_recent_exposure_count",
    "item_recent_positive_rate",
    "user_recent_activity",
    "user_category_affinity_score",
    "user_tag_affinity_score",
    "watch_ratio_mean_history",
    "watch_ratio_max_history",
]

LEAKY_OR_AUDIT_COLUMNS = {
    "label",
    "watch_ratio",
    "relevance_label",
    "future_watch_ratio",
    "test_label",
    "is_injected_positive",
    "is_retrieved_positive",
    "negative_type",
    "split",
}


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def build_feature_schema(
    frame: pd.DataFrame,
    source_names: list[str] | None = None,
    extra_feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    source_names = sorted(source_names if source_names is not None else infer_source_names(frame))
    columns = list(feature_columns(source_names))
    for column in OPTIONAL_NUMERIC_FEATURES:
        if column in frame.columns and column not in LEAKY_OR_AUDIT_COLUMNS:
            columns.append(column)
    for column in extra_feature_columns or []:
        if column not in columns and column not in LEAKY_OR_AUDIT_COLUMNS:
            columns.append(column)
    return {
        "version": SCHEMA_VERSION,
        "feature_family": "candidate_ltr_gbdt_compatible",
        "source_names": source_names,
        "feature_columns": columns,
        "excluded_columns": sorted(LEAKY_OR_AUDIT_COLUMNS),
    }


def make_lambdarank_features(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    source_names = [str(name) for name in schema.get("source_names", [])]
    columns = [str(column) for column in schema.get("feature_columns", [])]
    features = _candidate_features(frame, source_names)
    for column in columns:
        if column in features.columns:
            continue
        if column in LEAKY_OR_AUDIT_COLUMNS:
            features[column] = 0.0
            continue
        if column in frame.columns:
            features[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        else:
            features[column] = 0.0
    return features[columns].fillna(0.0)


def _candidate_features(frame: pd.DataFrame, source_names: list[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    source_rank = _numeric(frame, "source_rank", _numeric(frame, "rank", 0.0))
    rank = _numeric(frame, "rank", source_rank)
    item_popularity = _numeric(frame, "item_popularity", 0.0)
    user_activity = _numeric(frame, "user_activity_level", 0.0)
    features["retrieval_score"] = _numeric(frame, "retrieval_score", 0.0)
    features["source_rank"] = source_rank
    features["inverse_source_rank"] = 1.0 / np.maximum(source_rank.to_numpy(dtype=float), 1.0)
    features["rank"] = rank
    features["inverse_rank"] = 1.0 / np.maximum(rank.to_numpy(dtype=float), 1.0)
    features["source_count"] = _numeric(frame, "source_count", 0.0) if "source_count" in frame.columns else _source_count(frame)
    features["item_popularity"] = item_popularity
    features["log_item_popularity"] = np.log1p(np.maximum(item_popularity.to_numpy(dtype=float), 0.0))
    features["user_activity_level"] = user_activity
    features["log_user_activity_level"] = np.log1p(np.maximum(user_activity.to_numpy(dtype=float), 0.0))
    features["history_len"] = _numeric(frame, "history_len", 0.0)
    features["duration_bucket"] = _numeric(frame, "duration_bucket", 0.0)
    features["category_count"] = _count_nonzero_column(frame, "category_indices")
    features["tag_count"] = _count_nonzero_column(frame, "tag_indices")
    for column in [
        "category_overlap",
        "tag_overlap",
        "category_jaccard",
        "tag_jaccard",
        "history_category_count",
        "history_tag_count",
        "candidate_pool_size",
    ]:
        features[column] = _numeric(frame, column, 0.0)
    source_sets = None
    for source in source_names:
        if source_sets is None:
            source_sets = _source_sets(frame)
        features[f"source_has_{source}"] = [1.0 if source in values else 0.0 for values in source_sets]
    return features


def _numeric(frame: pd.DataFrame, column: str, fallback: Any) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if isinstance(fallback, pd.Series):
        return fallback.reindex(frame.index).fillna(0.0)
    return pd.Series(float(fallback), index=frame.index)


def _count_nonzero_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return frame[column].map(_nonzero_count).astype(float)


def _source_count(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([float(len(values)) for values in _source_sets(frame)], index=frame.index)


def _source_sets(frame: pd.DataFrame) -> list[set[str]]:
    if "retrieval_sources" in frame.columns:
        values = frame["retrieval_sources"]
    elif "retrieval_source" in frame.columns:
        values = frame["retrieval_source"]
    else:
        values = pd.Series("", index=frame.index)
    return [set(sources_from_value(value)) for value in values]


def _nonzero_count(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        return 1.0 if value else 0.0
    try:
        return float(sum(1 for item in value if int(item) != 0))
    except (TypeError, ValueError):
        return 0.0


def make_relevance_labels(frame: pd.DataFrame, label_mode: str = "graded") -> np.ndarray:
    label_values = frame["label"] if "label" in frame.columns else pd.Series(0, index=frame.index)
    labels = pd.to_numeric(label_values, errors="coerce").fillna(0.0).to_numpy()
    if label_mode == "binary":
        return (labels > 0).astype(np.int32)
    relevance = np.zeros(len(frame), dtype=np.int32)
    watch_ratio = None
    if "watch_ratio" in frame.columns:
        watch_ratio = pd.to_numeric(frame["watch_ratio"], errors="coerce").to_numpy()
    positive_mask = labels > 0
    relevance[positive_mask] = 2
    if watch_ratio is not None:
        partial_mask = (~positive_mask) & np.isfinite(watch_ratio) & (watch_ratio >= 0.3)
        complete_mask = positive_mask & np.isfinite(watch_ratio) & (watch_ratio >= 1.0)
        weak_positive_mask = positive_mask & np.isfinite(watch_ratio) & (watch_ratio < 1.0)
        relevance[partial_mask] = 1
        relevance[weak_positive_mask] = 1
        relevance[complete_mask] = 2
    return relevance


def grouped_ltr_frame(
    frame: pd.DataFrame,
    labels: np.ndarray | None = None,
    query_col: str = "user_id",
    min_group_size: int = 2,
    require_positive: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    if query_col not in frame.columns:
        raise ValueError(f"Missing query column for LambdaRank groups: {query_col}")
    labels = make_relevance_labels(frame) if labels is None else np.asarray(labels)
    work = frame.copy()
    work["_ltr_label"] = labels
    sort_columns = [query_col]
    for column in ["rank", "source_rank", "item_id"]:
        if column in work.columns:
            sort_columns.append(column)
    work = work.sort_values(sort_columns).reset_index(drop=True)

    kept_groups: list[pd.DataFrame] = []
    group_sizes: list[int] = []
    for _, group in work.groupby(query_col, sort=False):
        if len(group) < min_group_size:
            continue
        if require_positive and int((group["_ltr_label"] > 0).sum()) == 0:
            continue
        kept_groups.append(group)
        group_sizes.append(int(len(group)))
    if not kept_groups:
        return work.drop(columns=["_ltr_label"]).head(0), np.asarray([], dtype=np.int32), []
    grouped = pd.concat(kept_groups, ignore_index=True)
    grouped_labels = grouped.pop("_ltr_label").to_numpy(dtype=np.int32)
    return grouped, grouped_labels, group_sizes


def save_feature_schema(schema: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_lambdarank_model(model: Any, output_dir: str | Path, schema: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        booster.save_model(str(output / "model.txt"))
    save_feature_schema(schema, output / "feature_schema.json")


def load_lambdarank_model(model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    model_path = Path(model_dir) / "model.pkl"
    schema_path = Path(model_dir) / "feature_schema.json"
    if not model_path.exists() or not schema_path.exists():
        raise FileNotFoundError(f"Missing LambdaRank artifacts under {model_dir}")
    with model_path.open("rb") as file:
        model = pickle.load(file)
    return model, load_feature_schema(schema_path)


def predict_lambdarank_scores(model: Any, frame: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    features = make_lambdarank_features(frame, schema)
    scores = model.predict(features)
    return np.asarray(scores, dtype=np.float64)


def validation_ranking_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    cutoffs: list[int] | None = None,
    query_col: str = "user_id",
) -> dict[str, float]:
    cutoffs = cutoffs or [5, 10, 20, 50, 100]
    scored = frame[[query_col, "item_id"]].copy()
    scored["score"] = scores
    scored["label"] = make_relevance_labels(frame)
    rows: list[dict[str, float]] = []
    for _, group in scored.groupby(query_col):
        relevant = set(group[group["label"] > 0]["item_id"].astype(str))
        if not relevant:
            continue
        ranked = group.sort_values("score", ascending=False)["item_id"].astype(str).tolist()
        rows.append(_ranking_metrics(ranked, relevant, cutoffs))
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {f"val_{key}": float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


def _ranking_metrics(recommended: list[str], relevant: set[str], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        top = recommended[:k]
        hits = [1 if item_id in relevant else 0 for item_id in top]
        hit_count = sum(hits)
        metrics[f"Recall@{k}"] = hit_count / len(relevant) if relevant else 0.0
        metrics[f"NDCG@{k}"] = _ndcg(hits, min(len(relevant), k))
    metrics["MRR@10"] = _mrr(recommended[:10], relevant)
    metrics["MAP@10"] = _average_precision(recommended[:10], relevant)
    return metrics


def _ndcg(hits: list[int], ideal_hits: int) -> float:
    import math

    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _mrr(recommended: list[str], relevant: set[str]) -> float:
    for idx, item_id in enumerate(recommended, start=1):
        if item_id in relevant:
            return 1.0 / idx
    return 0.0


def _average_precision(recommended: list[str], relevant: set[str]) -> float:
    if not relevant or not recommended:
        return 0.0
    hits = 0
    total = 0.0
    for idx, item_id in enumerate(recommended, start=1):
        if item_id in relevant:
            hits += 1
            total += hits / idx
    return total / min(len(relevant), len(recommended))
