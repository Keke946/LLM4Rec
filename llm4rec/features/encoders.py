from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class IDEncoder:
    """Stable ID encoder with 0 reserved for unknown/padding."""

    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_idx: dict[str, int] = {"<UNK>": 0}
        self.idx_to_token: list[str] = ["<UNK>"]

    def fit(self, values: Iterable[object]) -> "IDEncoder":
        counts: dict[str, int] = {}
        for value in values:
            nested_values = _as_list(value)
            if nested_values is not None:
                for nested in nested_values:
                    counts[str(nested)] = counts.get(str(nested), 0) + 1
            else:
                counts[str(value)] = counts.get(str(value), 0) + 1
        for token in sorted(token for token, count in counts.items() if count >= self.min_freq):
            if token and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
        return self

    def transform_one(self, value: object) -> int:
        return self.token_to_idx.get(str(value), 0)

    def transform(self, values: Iterable[object]) -> list[int]:
        return [self.transform_one(value) for value in values]

    def transform_list(self, values: Iterable[object], max_len: int | None = None) -> list[int]:
        normalized = _as_list(values)
        if normalized is None:
            normalized = [values]
        encoded = [self.transform_one(value) for value in normalized]
        if max_len is not None:
            encoded = encoded[:max_len]
        return encoded

    def __len__(self) -> int:
        return len(self.idx_to_token)


@dataclass
class FeatureEncoders:
    user: IDEncoder
    item: IDEncoder
    author: IDEncoder
    category: IDEncoder
    tag: IDEncoder
    retrieval_source: IDEncoder
    duration_boundaries: list[float]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoders":
        with Path(path).open("rb") as file:
            encoders = pickle.load(file)
        if not hasattr(encoders, "retrieval_source"):
            encoders.retrieval_source = default_retrieval_source_encoder()
        else:
            encoders.retrieval_source.fit(default_retrieval_sources())
        return encoders


def default_retrieval_sources() -> list[str]:
    return [
        "unknown",
        "popular",
        "content_faiss",
        "exposure",
        "itemcf",
        "bpr",
        "two_tower",
        "lightgcn",
        "sasrec",
        "metadata",
        "trend",
    ]


def default_retrieval_source_encoder() -> IDEncoder:
    return IDEncoder().fit(default_retrieval_sources())


def bucketize_duration(value: object, boundaries: list[float]) -> int:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0
    if duration != duration:
        return 0
    for idx, boundary in enumerate(boundaries, start=1):
        if duration <= boundary:
            return idx
    return len(boundaries) + 1


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, str) or value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return None
