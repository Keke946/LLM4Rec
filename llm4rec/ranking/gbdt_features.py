from __future__ import annotations

import math
import ast
from typing import Any

import pandas as pd


BASE_GBDT_FEATURE_COLUMNS = [
    "retrieval_score",
    "source_rank",
    "inverse_source_rank",
    "rank",
    "inverse_rank",
    "source_count",
    "item_popularity",
    "log_item_popularity",
    "user_activity_level",
    "log_user_activity_level",
    "history_len",
    "duration_bucket",
    "category_count",
    "tag_count",
    "category_overlap",
    "tag_overlap",
    "category_jaccard",
    "tag_jaccard",
    "history_category_count",
    "history_tag_count",
    "candidate_pool_size",
]


def infer_source_names(frame: pd.DataFrame) -> list[str]:
    names: set[str] = set()
    for value in frame.get("retrieval_sources", []):
        names.update(sources_from_value(value))
    for value in frame.get("retrieval_source", []):
        names.update(sources_from_value(value))
    names.discard("")
    names.discard("unknown")
    return sorted(names)


def make_gbdt_features(frame: pd.DataFrame, source_names: list[str]) -> pd.DataFrame:
    # Share the vectorized implementation with LambdaRank; avoid a million-row
    # list of dictionaries when fitting the real-data candidate pool.
    from llm4rec.ranking.lambdarank import _candidate_features

    return _candidate_features(frame, source_names)[feature_columns(source_names)].fillna(0.0)


def feature_columns(source_names: list[str]) -> list[str]:
    return [*BASE_GBDT_FEATURE_COLUMNS, *[f"source_has_{name}" for name in source_names]]


def gbdt_feature_row(row: dict[str, Any], source_names: list[str]) -> dict[str, float]:
    source_rank = _float(row.get("source_rank", row.get("rank", 0.0)))
    rank = _float(row.get("rank", source_rank))
    item_popularity = _float(row.get("item_popularity", 0.0))
    user_activity = _float(row.get("user_activity_level", 0.0))
    sources = set(sources_from_value(row.get("retrieval_sources", row.get("retrieval_source", ""))))
    result = {
        "retrieval_score": _float(row.get("retrieval_score", 0.0)),
        "source_rank": source_rank,
        "inverse_source_rank": 1.0 / max(source_rank, 1.0),
        "rank": rank,
        "inverse_rank": 1.0 / max(rank, 1.0),
        "source_count": _float(row.get("source_count", len(sources))),
        "item_popularity": item_popularity,
        "log_item_popularity": math.log1p(max(item_popularity, 0.0)),
        "user_activity_level": user_activity,
        "log_user_activity_level": math.log1p(max(user_activity, 0.0)),
        "history_len": _float(row.get("history_len", 0.0)),
        "duration_bucket": _float(row.get("duration_bucket", 0.0)),
        "category_count": _nonzero_count(row.get("category_indices", [])),
        "tag_count": _nonzero_count(row.get("tag_indices", [])),
        "category_overlap": _float(row.get("category_overlap", 0.0)),
        "tag_overlap": _float(row.get("tag_overlap", 0.0)),
        "category_jaccard": _float(row.get("category_jaccard", 0.0)),
        "tag_jaccard": _float(row.get("tag_jaccard", 0.0)),
        "history_category_count": _float(row.get("history_category_count", 0.0)),
        "history_tag_count": _float(row.get("history_tag_count", 0.0)),
        "candidate_pool_size": _float(row.get("candidate_pool_size", 0.0)),
    }
    for source in source_names:
        result[f"source_has_{source}"] = 1.0 if source in sources else 0.0
    return result


def sources_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return sources_from_value(parsed)
        if "+" in value:
            return [part for part in value.split("+") if part]
        return [value]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(sources_from_value(item))
        return result
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return sources_from_value(converted)
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric


def _nonzero_count(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        return 1.0 if value else 0.0
    try:
        return float(sum(1 for item in value if int(item) != 0))
    except (TypeError, ValueError):
        return 0.0
