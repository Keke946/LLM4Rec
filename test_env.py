from __future__ import annotations

import os

from dotenv import load_dotenv


def mask(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


if __name__ == "__main__":
    load_dotenv()
    print("OPENAI_API_KEY:", mask(os.getenv("OPENAI_API_KEY")))
    print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "not set")
    print("OPENAI_MODEL:", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
