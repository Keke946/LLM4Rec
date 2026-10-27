from __future__ import annotations

import json
import ast
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetMeta:
    dataset: str
    num_users: int
    num_items: int
    num_interactions: int
    positive_rate: float
    split_stats: dict[str, int]
    item_fields: list[str]
    interaction_fields: list[str]
    is_subset: bool = False
    subset_info: dict[str, int | None] | None = None
    is_sample_fallback: bool = False
    data_origin: str = ""
    is_simulated: bool = False
    simulation_manifest: str | None = None
    split_strategy: str = "per_user_temporal"
    label_definition: str = ""


class BaseDatasetAdapter(ABC):
    """Adapter that maps raw public recommendation datasets to one canonical schema."""

    dataset_name = "base"

    def __init__(self, raw_dir: str | Path, **kwargs: Any) -> None:
        self.raw_dir = Path(raw_dir)
        self.kwargs = kwargs
        self.raw: dict[str, pd.DataFrame] = {}
        self.items: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.users: pd.DataFrame | None = None

    @abstractmethod
    def load_raw(self) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def build_items(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def build_interactions(self) -> pd.DataFrame:
        raise NotImplementedError

    def build_users(self) -> pd.DataFrame | None:
        return None

    def temporal_split(
        self,
        interactions: pd.DataFrame | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> pd.DataFrame:
        source = interactions if interactions is not None else self.interactions
        if source is None:
            raise ValueError("No interactions to split.")
        df = source.copy()
        if df.empty:
            df["split"] = []
            return df

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
        df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
        strategy = self.kwargs.get("split_strategy", "per_user_temporal")
        if strategy == "global_temporal":
            train_end = df["timestamp"].quantile(1 - val_ratio - test_ratio)
            val_end = df["timestamp"].quantile(1 - test_ratio)
            df["split"] = "test"
            df.loc[df["timestamp"] <= val_end, "split"] = "val"
            df.loc[df["timestamp"] <= train_end, "split"] = "train"
            return df
        if strategy != "per_user_temporal":
            raise ValueError(f"Unknown split strategy: {strategy}")
        splits: list[str] = []
        for _, group in df.groupby("user_id", sort=False):
            n = len(group)
            if n == 1:
                user_splits = ["train"]
            elif n == 2:
                user_splits = ["train", "test"]
            else:
                test_n = max(1, int(round(n * test_ratio)))
                val_n = max(1, int(round(n * val_ratio)))
                if test_n + val_n >= n:
                    test_n = 1
                    val_n = 1 if n >= 3 else 0
                train_n = n - val_n - test_n
                user_splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
            splits.extend(user_splits)
        df["split"] = splits
        return df

    def save_processed(self, output_dir: str | Path) -> DatasetMeta:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.raw = self.raw or self.load_raw()
        built_items = ensure_items_schema(self.build_items())
        built_interactions = self.build_interactions()
        built_items, built_interactions = self._apply_subset_sampling(built_items, built_interactions)

        self.items = ensure_items_schema(built_items)
        self.interactions = ensure_interactions_schema(self.temporal_split(built_interactions))
        self.users = self.build_users()
        if self.users is not None and "user_id" in self.users.columns:
            kept_users = set(self.interactions["user_id"].astype(str))
            self.users = self.users[self.users["user_id"].astype(str).isin(kept_users)].reset_index(drop=True)

        self.items.to_parquet(output / "items.parquet", index=False)
        self.interactions.to_parquet(output / "interactions.parquet", index=False)
        if self.users is not None:
            self.users.to_parquet(output / "users.parquet", index=False)

        meta = self.build_meta()
        with (output / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(meta), file, ensure_ascii=False, indent=2)
        return meta

    def build_meta(self) -> DatasetMeta:
        if self.items is None or self.interactions is None:
            raise ValueError("Call save_processed() or build items/interactions first.")
        split_stats = {
            str(split): int(count)
            for split, count in self.interactions["split"].value_counts().sort_index().items()
        }
        return DatasetMeta(
            dataset=self.dataset_name,
            num_users=int(self.interactions["user_id"].nunique()),
            num_items=int(self.items["item_id"].nunique()),
            num_interactions=int(len(self.interactions)),
            positive_rate=float(self.interactions["label"].mean()) if len(self.interactions) else 0.0,
            split_stats=split_stats,
            item_fields=list(self.items.columns),
            interaction_fields=list(self.interactions.columns),
            is_subset=bool(self._subset_info()),
            subset_info=self._subset_info() or None,
            is_sample_fallback=bool(self.kwargs.get("is_sample_fallback", False)),
            data_origin=str(self.kwargs.get("data_origin") or self.raw_dir),
            is_simulated=bool(self.kwargs.get("is_simulated", False)),
            simulation_manifest=self.kwargs.get("simulation_manifest"),
            split_strategy=str(self.kwargs.get("split_strategy", "per_user_temporal")),
            label_definition=(f"watch_ratio >= {self.kwargs.get('positive_watch_ratio_threshold', 1.0)}"
                              if self.dataset_name == "kuairec" else "dataset adapter threshold"),
        )

    def _subset_info(self) -> dict[str, int | str | None]:
        keys = [
            "sample_users",
            "sample_items",
            "max_interactions",
            "max_interactions_strategy",
            "max_interactions_seed",
        ]
        info = {key: self.kwargs.get(key) for key in keys if self.kwargs.get(key) is not None}
        return {
            key: int(value) if key != "max_interactions_strategy" and value is not None else value
            for key, value in info.items()
        }

    def _apply_subset_sampling(
        self,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_users = self.kwargs.get("sample_users")
        sample_items = self.kwargs.get("sample_items")
        max_interactions = self.kwargs.get("max_interactions")
        max_interactions_strategy = str(self.kwargs.get("max_interactions_strategy", "temporal"))
        max_interactions_seed = int(self.kwargs.get("max_interactions_seed", 42))
        if sample_users is None and sample_items is None and max_interactions is None:
            return items, interactions

        sampled = interactions.copy()
        if sample_users is not None:
            user_counts = sampled["user_id"].astype(str).value_counts()
            kept_users = set(user_counts.sort_values(ascending=False).head(int(sample_users)).index)
            sampled = sampled[sampled["user_id"].astype(str).isin(kept_users)]
        if sample_items is not None:
            item_counts = sampled["item_id"].astype(str).value_counts()
            kept_items = set(item_counts.sort_values(ascending=False).head(int(sample_items)).index)
            sampled = sampled[sampled["item_id"].astype(str).isin(kept_items)]
        if max_interactions is not None:
            max_n = int(max_interactions)
            if len(sampled) > max_n:
                if max_interactions_strategy == "random":
                    sampled = sampled.sample(n=max_n, random_state=max_interactions_seed)
                else:
                    sampled = sampled.sort_values(["timestamp", "user_id", "item_id"]).head(max_n)

        sampled = sampled.reset_index(drop=True)
        kept_item_ids = set(sampled["item_id"].astype(str))
        sampled_items = items[items["item_id"].astype(str).isin(kept_item_ids)].reset_index(drop=True)
        return sampled_items, sampled


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "(no genres listed)":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(part).strip() for part in parsed if str(part).strip()]
        except (SyntaxError, ValueError):
            pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def ensure_items_schema(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    for column in ["item_id", "title", "description", "author_id", "full_text"]:
        if column not in df.columns:
            df[column] = ""
    for column in ["categories", "tags"]:
        if column not in df.columns:
            df[column] = [[] for _ in range(len(df))]
        df[column] = df[column].apply(normalize_list)
    if "duration" not in df.columns:
        df["duration"] = pd.NA
    df["item_id"] = df["item_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    df["full_text"] = df["full_text"].fillna("").astype(str)
    missing_text = df["full_text"].str.len() == 0
    df.loc[missing_text, "full_text"] = df.loc[missing_text].apply(
        lambda row: (
            f"{row['title']}. {row['description']}. "
            f"{' '.join(row['categories'])} {' '.join(row['tags'])}"
        ),
        axis=1,
    )
    return df[
        ["item_id", "title", "description", "categories", "tags", "author_id", "duration", "full_text"]
        + [col for col in df.columns if col not in {
            "item_id", "title", "description", "categories", "tags", "author_id", "duration", "full_text"
        }]
    ].drop_duplicates("item_id").reset_index(drop=True)


def ensure_interactions_schema(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    for column in ["user_id", "item_id", "source"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    if "label" not in df.columns:
        df["label"] = 1
    if "watch_ratio" not in df.columns:
        df["watch_ratio"] = pd.NA
    if "impression_id" not in df.columns:
        df["impression_id"] = ""
    if "split" not in df.columns:
        df["split"] = "train"
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype("int64")
    df["split"] = df["split"].fillna("train").astype(str)
    return df[
        ["user_id", "item_id", "timestamp", "label", "watch_ratio", "impression_id", "source", "split"]
        + [col for col in df.columns if col not in {
            "user_id", "item_id", "timestamp", "label", "watch_ratio", "impression_id", "source", "split"
        }]
    ].reset_index(drop=True)
