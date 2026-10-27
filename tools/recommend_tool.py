from __future__ import annotations

from functools import lru_cache
from typing import Any

from llm4rec.recommender import LLM4RecRecommender


@lru_cache(maxsize=1)
def get_recommender() -> LLM4RecRecommender:
    return LLM4RecRecommender()


def recommend_videos(user_tags: list[str] | str, k: int = 5) -> list[dict[str, Any]]:
    """LangChain/tool-friendly recommendation function."""
    return get_recommender().recommend(user_tags, k=k)
