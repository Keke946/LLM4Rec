from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm4rec.datasets.base import BaseDatasetAdapter, normalize_list


class MovieLensAdapter(BaseDatasetAdapter):
    dataset_name = "movielens"

    def load_raw(self) -> dict[str, pd.DataFrame]:
        movies_path = _find_file(self.raw_dir, ["movies.csv"])
        ratings_path = _find_file(self.raw_dir, ["ratings.csv"])
        tags_path = _find_file(self.raw_dir, ["tags.csv"], required=False)

        raw = {
            "movies": pd.read_csv(movies_path),
            "ratings": pd.read_csv(ratings_path),
        }
        if tags_path is not None:
            raw["tags"] = pd.read_csv(tags_path)
        self.raw = raw
        return raw

    def build_items(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        movies = raw["movies"].copy()
        movies["item_id"] = movies["movieId"].astype(str)
        movies["title"] = movies["title"].fillna("").astype(str)
        movies["categories"] = movies["genres"].apply(normalize_list)

        tag_map: dict[str, list[str]] = {}
        if "tags" in raw:
            tags = raw["tags"].dropna(subset=["tag"]).copy()
            tags["movieId"] = tags["movieId"].astype(str)
            tag_map = tags.groupby("movieId")["tag"].apply(
                lambda values: sorted({str(value).strip().lower() for value in values if str(value).strip()})
            ).to_dict()

        movies["tags"] = movies["item_id"].map(tag_map).apply(lambda value: value if isinstance(value, list) else [])
        movies["description"] = ""
        movies["author_id"] = ""
        movies["duration"] = pd.NA
        movies["full_text"] = (
            movies["title"] + ". " + movies["categories"].apply(" ".join) + ". " + movies["tags"].apply(" ".join)
        )
        return movies[["item_id", "title", "description", "categories", "tags", "author_id", "duration", "full_text"]]

    def build_interactions(self) -> pd.DataFrame:
        raw = self.raw or self.load_raw()
        threshold = float(self.kwargs.get("positive_rating_threshold", 4.0))
        ratings = raw["ratings"].copy()
        ratings["user_id"] = ratings["userId"].astype(str)
        ratings["item_id"] = ratings["movieId"].astype(str)
        ratings["timestamp"] = pd.to_numeric(ratings.get("timestamp", 0), errors="coerce").fillna(0).astype("int64")
        ratings["label"] = (pd.to_numeric(ratings["rating"], errors="coerce").fillna(0) >= threshold).astype("int64")
        ratings["watch_ratio"] = pd.NA
        ratings["impression_id"] = ""
        ratings["source"] = self.dataset_name
        return ratings[["user_id", "item_id", "timestamp", "label", "watch_ratio", "impression_id", "source"]]


def _find_file(raw_dir: Path, names: list[str], required: bool = True) -> Path | None:
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
