from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from llm4rec.config import DEFAULT_PATHS


REQUIRED_VIDEO_COLUMNS = {"video_id", "title", "description", "tags", "labels"}


def parse_list_cell(value: Any) -> list[str]:
    """Parse CSV cells stored as Python-list-like strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = [part.strip() for part in text.split(",")]

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def load_videos(path: str | Path | None = None) -> pd.DataFrame:
    video_path = Path(path or DEFAULT_PATHS.video_csv)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video dataset not found: {video_path}. "
            "Run `python scripts/mini_youtube8m_generator.py` first."
        )

    df = pd.read_csv(video_path)
    missing = REQUIRED_VIDEO_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Video dataset is missing columns: {sorted(missing)}")

    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["tags"] = df["tags"].apply(parse_list_cell)
    df["labels"] = df["labels"].apply(parse_list_cell)
    df["full_text"] = (
        df["title"] + ". " + df["description"] + ". " + df["tags"].apply(" ".join)
    )
    return df


def load_user_behavior(path: str | Path | None = None) -> dict[str, dict[str, list[str]]]:
    behavior_path = Path(path or DEFAULT_PATHS.user_behavior_json)
    if not behavior_path.exists():
        raise FileNotFoundError(
            f"User behavior file not found: {behavior_path}. "
            "Run `python scripts/user_behavior_generator.py` first."
        )

    with behavior_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    normalized: dict[str, dict[str, list[str]]] = {}
    for user_id, info in raw.items():
        normalized[str(user_id)] = {
            "watched_videos": [str(video_id) for video_id in info.get("watched_videos", [])],
            "preferred_tags": [str(tag).strip().lower() for tag in info.get("preferred_tags", [])],
        }
    return normalized


def build_tag_binarizer(videos: pd.DataFrame) -> MultiLabelBinarizer:
    all_tags = sorted({tag for tags in videos["tags"] for tag in tags})
    binarizer = MultiLabelBinarizer(classes=all_tags)
    binarizer.fit(videos["tags"])
    return binarizer


def validate_video_dataset(videos: pd.DataFrame) -> dict[str, int]:
    duplicate_ids = int(videos["video_id"].duplicated().sum())
    empty_tags = int(videos["tags"].apply(lambda tags: len(tags) == 0).sum())
    empty_titles = int((videos["title"].str.len() == 0).sum())
    return {
        "rows": int(len(videos)),
        "duplicate_video_ids": duplicate_ids,
        "empty_tag_rows": empty_tags,
        "empty_title_rows": empty_titles,
        "unique_tags": int(len({tag for tags in videos["tags"] for tag in tags})),
    }
