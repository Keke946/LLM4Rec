from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4rec.config import DEFAULT_PATHS, ensure_project_dirs
from llm4rec.data import load_videos


def generate_user_behavior(
    users: int,
    watched_per_user: int,
    output: str,
    seed: int = 42,
) -> None:
    ensure_project_dirs()
    rng = random.Random(seed)
    videos = load_videos()
    all_tags = sorted({tag for tags in videos["tags"] for tag in tags})

    behavior: dict[str, dict[str, list[str]]] = {}
    for idx in range(1, users + 1):
        preferred_tags = rng.sample(all_tags, k=min(3, len(all_tags)))
        matched = videos[
            videos["tags"].apply(lambda tags: bool(set(preferred_tags) & set(tags)))
        ]["video_id"].tolist()
        pool = matched or videos["video_id"].tolist()
        watched = rng.sample(pool, k=min(watched_per_user, len(pool)))
        behavior[f"user_{idx}"] = {
            "watched_videos": watched,
            "preferred_tags": preferred_tags,
        }

    Path(output).write_text(
        json.dumps(behavior, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic user behavior data.")
    parser.add_argument("--users", type=int, default=20, help="Number of synthetic users.")
    parser.add_argument("--watched", type=int, default=8, help="Watched videos per user.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default=str(DEFAULT_PATHS.user_behavior_json), help="Output JSON path.")
    args = parser.parse_args()
    generate_user_behavior(args.users, args.watched, args.output, args.seed)
    print(f"Generated {args.users} users at {args.output}")


if __name__ == "__main__":
    main()
