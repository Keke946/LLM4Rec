from __future__ import annotations

import os
import re
from collections.abc import Sequence


def template_explanation(user_tags: Sequence[str], video_tags: Sequence[str], title: str) -> str:
    matched = [tag for tag in user_tags if tag in set(video_tags)]
    if matched:
        return f"你关注的 {', '.join(matched)} 与《{title}》的主题标签高度匹配。"
    if video_tags:
        return f"《{title}》覆盖 {', '.join(video_tags[:3])} 等内容，和你的兴趣方向相近。"
    return f"《{title}》和你的兴趣描述相近，因此推荐给你。"


def llm_explanation(
    user_tags: Sequence[str],
    video_tags: Sequence[str],
    title: str,
    allowed_item_ids: Sequence[str] | None = None,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    llm = ChatOpenAI(
        temperature=0.3,
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    prompt = (
        "请为已经由本地推荐系统选中的结果生成一句简洁、自然、可信的中文理由。\n"
        "禁止编造 item_id、分数、播放量、点赞数、用户行为或任何 metadata 中不存在的信息。\n"
        "如果需要提到 ID，只能使用允许列表中的 ID。\n"
        f"用户兴趣标签：{', '.join(user_tags)}\n"
        f"视频标题：{title}\n"
        f"视频标签：{', '.join(video_tags)}\n"
        f"允许的 item_id 列表：{', '.join(map(str, allowed_item_ids or []))}"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
    except Exception:
        return None


def generate_explanation(
    user_tags: Sequence[str],
    video_tags: Sequence[str],
    score: float,
    title: str,
    use_llm_fallback: bool = False,
    allowed_item_ids: Sequence[str] | None = None,
) -> str:
    fallback = template_explanation(user_tags, video_tags, title)
    if set(user_tags) & set(video_tags) or not use_llm_fallback:
        return fallback
    generated = llm_explanation(user_tags, video_tags, title, allowed_item_ids=allowed_item_ids)
    if not generated:
        return fallback
    return reject_hallucinated_item_ids(generated, allowed_item_ids or [], fallback)


ITEM_ID_PATTERN = re.compile(r"\b(?:vid_\d+|movie_\d+|item_\d+|N\d{2,}|\d{3,})\b", re.IGNORECASE)


def reject_hallucinated_item_ids(
    llm_text: str,
    allowed_item_ids: Sequence[str],
    fallback: str,
) -> str:
    """Return fallback if an LLM explanation mentions item IDs outside the candidate set."""
    allowed = {str(item_id) for item_id in allowed_item_ids}
    mentioned = set(ITEM_ID_PATTERN.findall(llm_text or ""))
    if any(item_id not in allowed for item_id in mentioned):
        return fallback
    return llm_text
