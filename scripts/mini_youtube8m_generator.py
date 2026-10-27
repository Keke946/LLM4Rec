from __future__ import annotations

import argparse
import csv
import sys
from itertools import cycle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4rec.config import DEFAULT_PATHS, ensure_project_dirs


VIDEO_TEMPLATES = [
    {
        "title": "Funny Cat Compilation",
        "description": "A compilation of funny cat videos.",
        "tags": ["funny", "cat", "compilation"],
        "labels": ["Animals", "Comedy"],
    },
    {
        "title": "Top Football Goals",
        "description": "The best football goals and tactical highlights.",
        "tags": ["football", "sports", "goals"],
        "labels": ["Sports"],
    },
    {
        "title": "Chocolate Cake Masterclass",
        "description": "Step-by-step guide to baking a chocolate cake.",
        "tags": ["baking", "cake", "chocolate"],
        "labels": ["Cooking"],
    },
    {
        "title": "Yoga for Beginners",
        "description": "A beginner-friendly yoga and mobility session.",
        "tags": ["yoga", "fitness", "beginner"],
        "labels": ["Health & Fitness"],
    },
    {
        "title": "Latest Tech Gadgets Review",
        "description": "Hands-on review of useful consumer tech gadgets.",
        "tags": ["tech", "gadgets", "review"],
        "labels": ["Technology"],
    },
    {
        "title": "Travel Vlog: Paris",
        "description": "Exploring food, museums, and streets in Paris.",
        "tags": ["travel", "vlog", "paris"],
        "labels": ["Travel"],
    },
    {
        "title": "Learn Python in 10 Minutes",
        "description": "Quick tutorial on Python basics and coding habits.",
        "tags": ["python", "tutorial", "coding"],
        "labels": ["Education"],
    },
    {
        "title": "Makeup Tutorial for Beginners",
        "description": "Simple makeup and beauty tips for everyday looks.",
        "tags": ["makeup", "beauty", "tutorial"],
        "labels": ["Beauty"],
    },
    {
        "title": "DIY Home Decor Ideas",
        "description": "Creative DIY projects for home decor upgrades.",
        "tags": ["diy", "home", "decor"],
        "labels": ["Lifestyle"],
    },
    {
        "title": "Epic Fails Compilation",
        "description": "A collection of lighthearted fail moments.",
        "tags": ["fails", "funny", "compilation"],
        "labels": ["Comedy"],
    },
]


def generate_dataset(num_rows: int, output: str) -> None:
    ensure_project_dirs()
    with open(output, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "title", "description", "tags", "labels"])
        for idx, template in zip(range(1, num_rows + 1), cycle(VIDEO_TEMPLATES)):
            season = (idx - 1) // len(VIDEO_TEMPLATES) + 1
            writer.writerow(
                [
                    f"vid_{idx:03d}",
                    f"{template['title']} #{season}",
                    template["description"],
                    template["tags"],
                    template["labels"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic mini video dataset.")
    parser.add_argument("--rows", type=int, default=200, help="Number of videos to generate.")
    parser.add_argument("--output", default=str(DEFAULT_PATHS.video_csv), help="Output CSV path.")
    args = parser.parse_args()
    generate_dataset(args.rows, args.output)
    print(f"Generated {args.rows} rows at {args.output}")


if __name__ == "__main__":
    main()
