from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+\-]+")

CHINESE_TAG_ALIASES = {
    "猫": "cat",
    "猫猫": "cat",
    "搞笑": "funny",
    "有趣": "funny",
    "合集": "compilation",
    "足球": "football",
    "体育": "sports",
    "进球": "goals",
    "烘焙": "baking",
    "蛋糕": "cake",
    "巧克力": "chocolate",
    "瑜伽": "yoga",
    "健身": "fitness",
    "入门": "beginner",
    "科技": "tech",
    "数码": "gadgets",
    "评测": "review",
    "旅行": "travel",
    "旅游": "travel",
    "巴黎": "paris",
    "编程": "coding",
    "代码": "coding",
    "python": "python",
    "教程": "tutorial",
    "美妆": "beauty",
    "化妆": "makeup",
    "家居": "home",
    "装饰": "decor",
    "手工": "diy",
    "翻车": "fails",
}


def normalize_tags(tags: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if not value:
            continue
        value = CHINESE_TAG_ALIASES.get(value, value)
        if value not in normalized:
            normalized.append(value)
    return normalized


def extract_interest_tags(
    user_input: str | Iterable[str],
    vocabulary: Iterable[str] | None = None,
) -> list[str]:
    """Extract normalized interest tags from free text or an existing tag list."""
    vocab = {tag.lower() for tag in vocabulary} if vocabulary is not None else None

    if not isinstance(user_input, str):
        tags = normalize_tags(user_input)
        if vocab:
            matched = [tag for tag in tags if tag in vocab]
            return matched or tags
        return tags

    text = user_input.strip().lower()
    tags: list[str] = []

    for alias, canonical in CHINESE_TAG_ALIASES.items():
        if alias.lower() in text and canonical not in tags:
            tags.append(canonical)

    for token in TOKEN_RE.findall(text):
        canonical = CHINESE_TAG_ALIASES.get(token, token)
        if canonical not in tags:
            tags.append(canonical)

    if vocab:
        matched = [tag for tag in tags if tag in vocab]
        return matched or tags
    return tags
