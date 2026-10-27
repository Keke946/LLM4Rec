from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from tools.recommend_tool import recommend_videos


load_dotenv()


def _recommend_for_agent(user_input: str) -> str:
    results = recommend_videos(user_input, k=5)
    return json.dumps(results, ensure_ascii=False, indent=2)


def get_recommender_agent():
    """Create the LangChain agent lazily so offline scripts can import safely."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Use `python run_chat.py --offline` "
            "for the local recommender demo, or set an API key for the LLM agent."
        )

    try:
        from langchain.agents import Tool, initialize_agent
        from langchain.agents.agent_types import AgentType
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain dependencies are missing. Run `pip install -r requirements.txt`."
        ) from exc

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )

    tools = [
        Tool.from_function(
            func=_recommend_for_agent,
            name="VideoRecommender",
            description=(
                "Recommend videos from a natural-language interest request. "
                "Input should be the user's original request, such as "
                "'我想看猫猫搞笑视频' or 'python tutorial'."
            ),
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
    )
