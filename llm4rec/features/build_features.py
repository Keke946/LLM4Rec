from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from llm4rec.features.encoders import (
    FeatureEncoders,
    IDEncoder,
    bucketize_duration,
    default_retrieval_source_encoder,
)


def build_ranking_features(
    processed_dir: str | Path,
    output_dir: str | Path,
    max_history_len: int = 20,
) -> FeatureEncoders:
    processed = Path(processed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed / "items.parquet")
    interactions = pd.read_parquet(processed / "interactions.parquet")
    item_features = _prepare_item_features(items)

    encoders = FeatureEncoders(
        user=IDEncoder().fit(interactions["user_id"]),
        item=IDEncoder().fit(items["item_id"]),
        author=IDEncoder().fit(items.get("author_id", pd.Series([""] * len(items)))),
        category=IDEncoder().fit(items["categories"]),
        tag=IDEncoder().fit(items["tags"]),
        retrieval_source=default_retrieval_source_encoder(),
        duration_boundaries=[10, 30, 60, 120, 300, 600, 1200],
    )
    encoders.save(output / "encoders.pkl")

    item_lookup = _encode_item_lookup(item_features, encoders)
    samples = _build_samples(interactions, item_lookup, encoders, max_history_len)

    for split in ["train", "val", "test"]:
        split_samples = samples[samples["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        split_samples.to_parquet(output / f"{split}_rank_samples.parquet", index=False)
    return encoders


class RankingSampleDataset(Dataset):
    def __init__(self, samples_path: str | Path, max_history_len: int = 20) -> None:
        self.samples = pd.read_parquet(samples_path).reset_index(drop=True)
        self.max_history_len = max_history_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.samples.iloc[index]
        history = _pad_list(row.get("history_item_indices", []), self.max_history_len)
        return {
            "user_idx": torch.tensor(int(row["user_idx"]), dtype=torch.long),
            "item_idx": torch.tensor(int(row["item_idx"]), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "author_idx": torch.tensor(int(row.get("author_idx", 0)), dtype=torch.long),
            "duration_bucket": torch.tensor(int(row.get("duration_bucket", 0)), dtype=torch.long),
            "category_indices": torch.tensor(_pad_list(row.get("category_indices", []), 8), dtype=torch.long),
            "tag_indices": torch.tensor(_pad_list(row.get("tag_indices", []), 16), dtype=torch.long),
            "history_item_indices": torch.tensor(history, dtype=torch.long),
            "retrieval_source_indices": torch.tensor(
                _pad_list(row.get("retrieval_source_indices", []), 8),
                dtype=torch.long,
            ),
            "retrieval_score": torch.tensor(float(row.get("retrieval_score", 0.0)), dtype=torch.float32),
            "source_rank": torch.tensor(float(row.get("source_rank", 0.0)), dtype=torch.float32),
            "source_count": torch.tensor(float(row.get("source_count", 0.0)), dtype=torch.float32),
            "item_popularity": torch.tensor(float(row.get("item_popularity", 0.0)), dtype=torch.float32),
            "user_activity_level": torch.tensor(float(row.get("user_activity_level", 0.0)), dtype=torch.float32),
            "history_len": torch.tensor(float(row.get("history_len", len([x for x in history if x]))), dtype=torch.float32),
            "dense_features": torch.tensor(
                [
                    float(row.get("retrieval_score", 0.0)),
                    float(row.get("source_rank", 0.0)),
                    float(row.get("source_count", 0.0)),
                    float(row.get("item_popularity", 0.0)),
                    float(row.get("user_activity_level", 0.0)),
                    float(row.get("history_len", len([x for x in history if x]))),
                ],
                dtype=torch.float32,
            ),
        }


def _prepare_item_features(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    for column in ["categories", "tags"]:
        if column not in df.columns:
            df[column] = [[] for _ in range(len(df))]
    if "author_id" not in df.columns:
        df["author_id"] = ""
    if "duration" not in df.columns:
        df["duration"] = pd.NA
    return df


def _encode_item_lookup(items: pd.DataFrame, encoders: FeatureEncoders) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        lookup[item_id] = {
            "item_idx": encoders.item.transform_one(item_id),
            "author_idx": encoders.author.transform_one(row.get("author_id", "")),
            "category_indices": encoders.category.transform_list(row.get("categories", []), max_len=8),
            "tag_indices": encoders.tag.transform_list(row.get("tags", []), max_len=16),
            "duration_bucket": bucketize_duration(row.get("duration"), encoders.duration_boundaries),
        }
    return lookup


def _build_samples(
    interactions: pd.DataFrame,
    item_lookup: dict[str, dict[str, Any]],
    encoders: FeatureEncoders,
    max_history_len: int,
) -> pd.DataFrame:
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    train_positive_counts = interactions[
        (interactions["split"] == "train") & (interactions["label"] == 1)
    ]
    item_popularity = train_positive_counts.groupby("item_id").size().to_dict()
    user_activity = train_positive_counts.groupby("user_id").size().to_dict()
    train_history_by_user: dict[str, list[int]] = {}
    val_history_by_user: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []

    for _, row in interactions.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        split = str(row["split"])
        item_info = item_lookup.get(
            item_id,
            {
                "item_idx": 0,
                "author_idx": 0,
                "category_indices": [],
                "tag_indices": [],
                "duration_bucket": 0,
            },
        )
        train_history = train_history_by_user.get(user_id, [])
        val_history = val_history_by_user.get(user_id, [])
        if split == "train":
            history = train_history[-max_history_len:]
        elif split == "val":
            history = (train_history + val_history)[-max_history_len:]
        else:
            history = (train_history + val_history)[-max_history_len:]
        rows.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "user_idx": encoders.user.transform_one(user_id),
                "item_idx": item_info["item_idx"],
                "label": int(row["label"]),
                "category_indices": item_info["category_indices"],
                "tag_indices": item_info["tag_indices"],
                "author_idx": item_info["author_idx"],
                "duration_bucket": item_info["duration_bucket"],
                "history_item_indices": history,
                "history_len": len(history),
                "retrieval_score": 0.0,
                "retrieval_source_indices": [encoders.retrieval_source.transform_one("unknown")],
                "source_rank": 0.0,
                "source_count": 0.0,
                "item_popularity": float(item_popularity.get(item_id, 0)),
                "user_activity_level": float(user_activity.get(user_id, 0)),
                "timestamp": int(row["timestamp"]),
                "split": split,
            }
        )
        if int(row["label"]) == 1:
            if split == "train":
                train_history_by_user.setdefault(user_id, []).append(item_info["item_idx"])
            elif split == "val":
                val_history_by_user.setdefault(user_id, []).append(item_info["item_idx"])

    return pd.DataFrame(rows)


def _pad_list(values: object, length: int) -> list[int]:
    if not isinstance(values, list):
        if values is None:
            values = []
        elif hasattr(values, "tolist"):
            values = values.tolist()
        else:
            values = list(values)
    result = [int(value) for value in values[:length]]
    return result + [0] * (length - len(result))
