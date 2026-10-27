from __future__ import annotations

from collections import Counter, defaultdict


def apply_business_reranker(
    item_ids: list[str],
    item_meta: dict[str, dict[str, object]],
    popularity: Counter[str],
    tail_items: set[str],
    mode: str = "none",
    top_k: int | None = None,
) -> list[str]:
    if mode in {"", "none", None}:
        return item_ids[:top_k] if top_k is not None else list(item_ids)
    selected: list[str] = []
    category_counts: dict[str, int] = defaultdict(int)
    author_counts: dict[str, int] = defaultdict(int)
    remaining = list(item_ids)
    target_k = top_k or len(item_ids)
    while remaining and len(selected) < target_k:
        scored = []
        for position, item_id in enumerate(remaining):
            meta = item_meta.get(item_id, {})
            categories = _as_list(meta.get("categories", []))
            author_id = str(meta.get("author_id", ""))
            score = -float(position)
            if mode in {"diversity", "diversity_tail"}:
                score += _diversity_bonus(categories, category_counts)
                score -= 0.25 * author_counts.get(author_id, 0)
                if _category_cap_exceeded(categories, category_counts, cap=2):
                    score -= 2.0
            if mode == "diversity_tail":
                if item_id in tail_items:
                    score += 0.35
                score += 0.08 / max(1.0, float(popularity.get(item_id, 0) + 1))
            scored.append((score, position, item_id))
        _, _, chosen = max(scored, key=lambda row: (row[0], -row[1]))
        selected.append(chosen)
        remaining.remove(chosen)
        chosen_meta = item_meta.get(chosen, {})
        for category in _as_list(chosen_meta.get("categories", [])):
            category_counts[category] += 1
        author_counts[str(chosen_meta.get("author_id", ""))] += 1
    return selected


def _as_list(value: object) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    return [str(value)] if str(value) else []


def _diversity_bonus(categories: list[str], category_counts: dict[str, int]) -> float:
    if not categories:
        return 0.0
    return sum(1.0 / (1.0 + category_counts.get(category, 0)) for category in categories) / len(categories)


def _category_cap_exceeded(categories: list[str], category_counts: dict[str, int], cap: int) -> bool:
    return any(category_counts.get(category, 0) >= cap for category in categories)
