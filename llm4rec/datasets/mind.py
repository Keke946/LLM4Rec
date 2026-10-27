from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm4rec.datasets.base import BaseDatasetAdapter


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


class MINDAdapter(BaseDatasetAdapter):
    dataset_name = "mind"

    def load_raw(self) -> dict[str, pd.DataFrame]:
        news_path = _find_file(self.raw_dir, "news.tsv")
        behaviors_path = _find_file(self.raw_dir, "behaviors.tsv")
        raw = {
            "news": pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS),
            "behaviors": pd.read_csv(behaviors_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS),
        }
        self.raw = raw
        return raw

    def build_items(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        news = raw["news"].copy()
        news["item_id"] = news["news_id"].astype(str)
        news["title"] = news["title"].fillna("").astype(str)
        news["description"] = news["abstract"].fillna("").astype(str)
        news["categories"] = news.apply(
            lambda row: [value for value in [str(row.get("category", "")).strip(), str(row.get("subcategory", "")).strip()] if value and value != "nan"],
            axis=1,
        )
        news["tags"] = news.apply(_extract_entity_tags, axis=1)
        news["author_id"] = news["category"].fillna("").astype(str)
        news["duration"] = pd.NA
        news["full_text"] = (
            news["title"]
            + ". "
            + news["description"]
            + ". "
            + news["categories"].apply(" ".join)
            + " "
            + news["tags"].apply(" ".join)
        )
        return news[["item_id", "title", "description", "categories", "tags", "author_id", "duration", "full_text"]]

    def build_interactions(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        rows: list[dict[str, object]] = []
        for _, row in raw["behaviors"].iterrows():
            timestamp = _parse_mind_time(row["time"])
            impression_id = str(row["impression_id"])
            user_id = str(row["user_id"])
            for impression in str(row["impressions"]).split():
                if "-" not in impression:
                    continue
                item_id, label = impression.rsplit("-", 1)
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "label": int(label),
                        "watch_ratio": pd.NA,
                        "impression_id": impression_id,
                        "source": self.dataset_name,
                    }
                )
        return pd.DataFrame(rows)

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
        df["_group_key"] = df["user_id"].astype(str) + "::" + df["impression_id"].astype(str)
        groups = (
            df.groupby(["user_id", "_group_key"], as_index=False)["timestamp"]
            .min()
            .sort_values(["user_id", "timestamp", "_group_key"])
            .reset_index(drop=True)
        )
        group_split: dict[str, str] = {}
        for _, user_groups in groups.groupby("user_id", sort=False):
            n = len(user_groups)
            if n == 1:
                splits = ["train"]
            elif n == 2:
                splits = ["train", "test"]
            else:
                test_n = max(1, int(round(n * test_ratio)))
                val_n = max(1, int(round(n * val_ratio)))
                if test_n + val_n >= n:
                    test_n = 1
                    val_n = 1 if n >= 3 else 0
                train_n = n - val_n - test_n
                splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
            for key, split in zip(user_groups["_group_key"], splits):
                group_split[str(key)] = split
        df["split"] = df["_group_key"].map(group_split).fillna("train")
        return df.drop(columns=["_group_key"]).reset_index(drop=True)

    def build_users(self) -> pd.DataFrame | None:
        raw = self.raw or self.load_raw()
        users = raw["behaviors"][["user_id", "history"]].copy()
        users["user_id"] = users["user_id"].astype(str)
        users["history_item_ids"] = users["history"].fillna("").astype(str).apply(lambda text: [item for item in text.split() if item])
        return users.groupby("user_id", as_index=False)["history_item_ids"].last()


def _find_file(raw_dir: Path, name: str) -> Path:
    direct = raw_dir / name
    if direct.exists():
        return direct
    matches = list(raw_dir.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {raw_dir}")


def _parse_mind_time(value: object) -> int:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return 0
    return int(parsed.timestamp())


def _extract_entity_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    for column in ["title_entities", "abstract_entities"]:
        text = row.get(column)
        if pd.isna(text) or not str(text).strip():
            continue
        try:
            entities = json.loads(str(text))
        except json.JSONDecodeError:
            entities = []
        for entity in entities:
            label = entity.get("Label") or entity.get("WikidataId")
            if label and str(label) not in tags:
                tags.append(str(label))
    return tags
