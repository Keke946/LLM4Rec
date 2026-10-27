from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from llm4rec.datasets.base import BaseDatasetAdapter, normalize_list


class KuaiRecAdapter(BaseDatasetAdapter):
    dataset_name = "kuairec"

    def load_raw(self) -> dict[str, pd.DataFrame]:
        matrix = str(self.kwargs.get("kuairec_matrix", "auto")).lower()
        if matrix == "big":
            interaction_names = ["big_matrix.csv"]
        elif matrix == "small":
            interaction_names = ["small_matrix.csv"]
        else:
            interaction_names = ["small_matrix.csv", "big_matrix.csv", "interactions.csv", "ratings.csv"]
        interaction_path = _first_existing(
            self.raw_dir,
            interaction_names,
        )
        item_path = _first_existing(
            self.raw_dir,
            ["item_categories.csv", "video_features_basic_pure.csv", "video_features.csv", "items.csv"],
            required=False,
        )
        # The public big matrix is >12M rows. Avoid loading duplicate human-readable
        # date strings and duration columns when the official ratio is present.
        columns = list(pd.read_csv(interaction_path, nrows=0).columns)
        usecols = ([c for c in ["user_id", "video_id", "timestamp", "watch_ratio"] if c in columns]
                   if {"user_id", "video_id", "timestamp", "watch_ratio"}.issubset(columns) else None)
        raw = {"interactions": pd.read_csv(interaction_path, usecols=usecols)}
        if item_path is not None:
            raw["items"] = pd.read_csv(item_path)
        self.raw = raw
        return raw

    def build_items(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        if "items" in raw:
            items = raw["items"].copy()
        else:
            unique_ids = sorted(raw["interactions"][_find_col(raw["interactions"], ["video_id", "item_id"])].astype(str).unique())
            items = pd.DataFrame({"video_id": unique_ids})

        video_col = _find_col(items, ["video_id", "item_id", "photo_id"])
        title_col = _find_optional_col(items, ["title", "caption", "video_title"])
        category_col = _find_optional_col(items, ["category", "category_name", "first_level_category_name", "second_level_category_name", "feat"])
        tag_col = _find_optional_col(items, ["tag", "tags", "manual_tag_list", "feat"])
        author_col = _find_optional_col(items, ["author_id", "user_id", "creator_id"])
        duration_col = _find_optional_col(items, ["duration", "video_duration"])

        result = pd.DataFrame()
        result["item_id"] = items[video_col].astype(str)
        result["title"] = items[title_col].fillna("").astype(str) if title_col else "video_" + result["item_id"]
        result["description"] = ""
        result["categories"] = items[category_col].apply(normalize_list) if category_col else [[] for _ in range(len(items))]
        result["tags"] = items[tag_col].apply(normalize_list) if tag_col else result["categories"]
        result["author_id"] = items[author_col].fillna("").astype(str) if author_col else ""
        result["duration"] = pd.to_numeric(items[duration_col], errors="coerce") if duration_col else pd.NA
        result["full_text"] = (
            result["title"].astype(str)
            + ". "
            + result["categories"].apply(" ".join)
            + " "
            + result["tags"].apply(" ".join)
        )
        return result

    def build_interactions(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        df = raw["interactions"].copy()
        threshold = float(self.kwargs.get("positive_watch_ratio_threshold", 1.0))

        user_col = _find_col(df, ["user_id", "uid"])
        item_col = _find_col(df, ["video_id", "item_id", "photo_id"])
        timestamp_col = _find_optional_col(df, ["timestamp", "time", "date", "request_timestamp"])
        play_col = _find_optional_col(df, ["play_duration", "play_time", "watch_duration"])
        duration_col = _find_optional_col(df, ["video_duration", "duration"])
        watch_ratio_col = _find_optional_col(df, ["watch_ratio", "play_ratio"])

        result = pd.DataFrame()
        result["user_id"] = df[user_col].astype(str)
        result["item_id"] = df[item_col].astype(str)
        if timestamp_col:
            result["timestamp"] = _parse_timestamp(df[timestamp_col])
        else:
            result["timestamp"] = range(len(df))

        if watch_ratio_col:
            watch_ratio = pd.to_numeric(df[watch_ratio_col], errors="coerce")
        elif play_col and duration_col:
            play = pd.to_numeric(df[play_col], errors="coerce").fillna(0)
            duration = pd.to_numeric(df[duration_col], errors="coerce").replace(0, pd.NA)
            watch_ratio = (play / duration).fillna(0)
        else:
            watch_ratio = pd.Series([1.0] * len(df))

        result["watch_ratio"] = watch_ratio.astype("float64")
        result["label"] = (result["watch_ratio"] >= threshold).astype("int64")
        result["impression_id"] = ""
        result["source"] = self.dataset_name
        return result[["user_id", "item_id", "timestamp", "label", "watch_ratio", "impression_id", "source"]]


def _first_existing(raw_dir: Path, names: list[str], required: bool = True) -> Path | None:
    for name in names:
        direct = raw_dir / name
        if direct.exists():
            return direct
        matches = list(raw_dir.rglob(name))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find one of {names} under {raw_dir}")
    return None


def _parse_timestamp(values: pd.Series) -> pd.Series:
    """Parse KuaiRec timestamps without treating Unix seconds as nanoseconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    if is_numeric_dtype(values) or numeric.notna().mean() >= 0.9:
        finite = numeric.dropna()
        if finite.empty:
            return pd.Series([0] * len(values), index=values.index, dtype="int64")
        median = float(finite.abs().median())
        if 19_000_000 <= median <= 21_000_000:
            parsed = pd.to_datetime(
                numeric.fillna(19700101).astype("int64").astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
            return (parsed.fillna(pd.Timestamp("1970-01-01")).astype("int64") // 10**9).astype("int64")
        if median >= 1e17:
            numeric = numeric / 1e9
        elif median >= 1e14:
            numeric = numeric / 1e6
        elif median >= 1e11:
            numeric = numeric / 1e3
        return numeric.fillna(0).astype("int64")

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return (parsed.fillna(pd.Timestamp("1970-01-01")).astype("int64") // 10**9).astype("int64")
    return numeric.fillna(0).astype("int64")


def _find_col(df: pd.DataFrame, names: list[str]) -> str:
    col = _find_optional_col(df, names)
    if col is None:
        raise ValueError(f"Could not find required column among {names}. Existing columns: {list(df.columns)}")
    return col


def _find_optional_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None
