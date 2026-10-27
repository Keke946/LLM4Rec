from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CATEGORIES = [
    "sports",
    "music",
    "food",
    "game",
    "travel",
    "education",
    "comedy",
    "fashion",
    "beauty",
    "pets",
    "technology",
    "movie",
    "finance",
    "fitness",
    "life",
    "auto",
    "parenting",
    "news",
    "photography",
    "dance",
]

TAGS_BY_CATEGORY = {
    "sports": ["basketball", "football", "workout", "outdoor"],
    "music": ["guitar", "cover", "playlist", "concert"],
    "food": ["recipe", "street_food", "baking", "healthy"],
    "game": ["mobile_game", "walkthrough", "esports", "strategy"],
    "travel": ["citywalk", "vlog", "hotel", "nature"],
    "education": ["math", "language", "career", "science"],
    "comedy": ["sketch", "reaction", "daily_fun", "meme"],
    "fashion": ["outfit", "streetwear", "shopping", "trend"],
    "beauty": ["makeup", "skincare", "hair", "review"],
    "pets": ["cat", "dog", "cute", "training"],
    "technology": ["ai", "phone", "coding", "gadget"],
    "movie": ["trailer", "review", "clip", "behind_scene"],
    "finance": ["stock", "saving", "macro", "career"],
    "fitness": ["yoga", "running", "muscle", "diet"],
    "life": ["home", "routine", "family", "study"],
    "auto": ["ev", "test_drive", "maintenance", "review"],
    "parenting": ["baby", "school", "family", "health"],
    "news": ["local", "society", "world", "policy"],
    "photography": ["camera", "editing", "portrait", "landscape"],
    "dance": ["hiphop", "practice", "stage", "challenge"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KuaiRec-compatible simulated raw data.")
    parser.add_argument("--output_dir", default="data/raw/kuairec")
    parser.add_argument("--num_users", type=int, default=1000)
    parser.add_argument("--num_items", type=int, default=5000)
    parser.add_argument("--num_interactions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_users <= 0 or args.num_items <= 0 or args.num_interactions <= 0:
        raise ValueError("num_users, num_items and num_interactions must be positive.")
    if args.num_interactions < args.num_items:
        raise ValueError("num_interactions must be >= num_items so every item can appear at least once.")
    if args.num_interactions < args.num_users:
        raise ValueError("num_interactions must be >= num_users so every user can appear at least once.")

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output} is not empty. Re-run with --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    item_meta = _build_items(args.num_items, rng)
    user_profiles = _build_user_profiles(args.num_users, rng)
    _write_items(output / "item_categories.csv", item_meta)
    _write_interactions(
        output / "big_matrix.csv",
        item_meta=item_meta,
        user_profiles=user_profiles,
        num_interactions=args.num_interactions,
        rng=rng,
    )
    manifest = {
        "dataset": "kuairec",
        "is_simulated": True,
        "schema_compatible_with": "KuaiRecAdapter",
        "num_users": args.num_users,
        "num_items": args.num_items,
        "num_interactions": args.num_interactions,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "interactions": "big_matrix.csv",
            "items": "item_categories.csv",
        },
        "warning": "Synthetic data for local pipeline validation only. Do not report it as real KuaiRec results.",
    }
    (output / "simulated_data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Generated simulated KuaiRec raw data at {output}: "
        f"users={args.num_users} items={args.num_items} interactions={args.num_interactions}"
    )


def _build_items(num_items: int, rng: random.Random) -> dict[str, dict[str, object]]:
    items: dict[str, dict[str, object]] = {}
    author_count = max(100, num_items // 12)
    for idx in range(1, num_items + 1):
        item_id = f"v_{idx:06d}"
        category = CATEGORIES[(idx - 1) % len(CATEGORIES)]
        tags = rng.sample(TAGS_BY_CATEGORY[category], k=2)
        duration = rng.randint(8, 900)
        items[item_id] = {
            "video_id": item_id,
            "category": category,
            "tags": "|".join([category, *tags]),
            "author_id": f"a_{rng.randint(1, author_count):05d}",
            "video_duration": duration,
            "title": f"{category} video {idx:06d}",
        }
    return items


def _build_user_profiles(num_users: int, rng: random.Random) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for idx in range(1, num_users + 1):
        user_id = f"u_{idx:05d}"
        interests = rng.sample(CATEGORIES, k=3)
        profiles[user_id] = {
            "interests": interests,
            "engagement": rng.uniform(-0.15, 0.25),
        }
    return profiles


def _write_items(path: Path, item_meta: dict[str, dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video_id", "category", "tags", "author_id", "video_duration", "title"],
        )
        writer.writeheader()
        for row in item_meta.values():
            writer.writerow(row)


def _write_interactions(
    path: Path,
    item_meta: dict[str, dict[str, object]],
    user_profiles: dict[str, dict[str, object]],
    num_interactions: int,
    rng: random.Random,
) -> None:
    item_ids = list(item_meta)
    user_ids = list(user_profiles)
    items_by_category: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    for item_id, meta in item_meta.items():
        items_by_category[str(meta["category"])].append(item_id)

    target_counts = _target_counts(len(user_ids), num_interactions)
    user_seen: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    user_offsets: dict[str, int] = {user_id: idx for idx, user_id in enumerate(user_ids)}
    base_ts = 1_704_067_200
    rows_written = 0

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "user_id",
                "video_id",
                "timestamp",
                "play_duration",
                "video_duration",
                "watch_ratio",
            ],
        )
        writer.writeheader()

        for item_idx, item_id in enumerate(item_ids):
            user_id = user_ids[item_idx % len(user_ids)]
            _write_one_interaction(
                writer,
                user_id=user_id,
                item_id=item_id,
                item_meta=item_meta,
                user_profiles=user_profiles,
                user_seen=user_seen,
                event_index=len(user_seen[user_id]),
                timestamp=base_ts + user_offsets[user_id] + len(user_seen[user_id]) * 600,
                rng=rng,
            )
            rows_written += 1

        user_cursor = 0
        while rows_written < num_interactions:
            user_id = user_ids[user_cursor % len(user_ids)]
            user_cursor += 1
            if len(user_seen[user_id]) >= target_counts[user_id]:
                continue
            item_id = _sample_item_for_user(
                user_id,
                item_ids,
                items_by_category,
                user_profiles,
                user_seen,
                rng,
            )
            _write_one_interaction(
                writer,
                user_id=user_id,
                item_id=item_id,
                item_meta=item_meta,
                user_profiles=user_profiles,
                user_seen=user_seen,
                event_index=len(user_seen[user_id]),
                timestamp=base_ts + user_offsets[user_id] + len(user_seen[user_id]) * 600,
                rng=rng,
            )
            rows_written += 1


def _target_counts(num_users: int, num_interactions: int) -> dict[str, int]:
    base = num_interactions // num_users
    remainder = num_interactions % num_users
    return {
        f"u_{idx:05d}": base + int(idx <= remainder)
        for idx in range(1, num_users + 1)
    }


def _sample_item_for_user(
    user_id: str,
    item_ids: list[str],
    items_by_category: dict[str, list[str]],
    user_profiles: dict[str, dict[str, object]],
    user_seen: dict[str, set[str]],
    rng: random.Random,
) -> str:
    interests = list(user_profiles[user_id]["interests"])
    for _ in range(30):
        if rng.random() < 0.82:
            category = rng.choice(interests)
            item_id = rng.choice(items_by_category[category])
        else:
            item_id = rng.choice(item_ids)
        if item_id not in user_seen[user_id]:
            return item_id
    unseen = [item_id for item_id in item_ids if item_id not in user_seen[user_id]]
    return rng.choice(unseen or item_ids)


def _write_one_interaction(
    writer: csv.DictWriter,
    user_id: str,
    item_id: str,
    item_meta: dict[str, dict[str, object]],
    user_profiles: dict[str, dict[str, object]],
    user_seen: dict[str, set[str]],
    event_index: int,
    timestamp: int,
    rng: random.Random,
) -> None:
    profile = user_profiles[user_id]
    meta = item_meta[item_id]
    duration = int(meta["video_duration"])
    affinity = str(meta["category"]) in set(profile["interests"])
    mean_ratio = 1.16 if affinity else 0.64
    ratio = rng.gauss(mean_ratio + float(profile["engagement"]), 0.32)
    if event_index % 17 == 0:
        ratio += 0.28
    watch_ratio = max(0.05, min(2.8, ratio))
    play_duration = max(1, int(round(duration * watch_ratio)))
    writer.writerow(
        {
            "user_id": user_id,
            "video_id": item_id,
            "timestamp": timestamp,
            "play_duration": play_duration,
            "video_duration": duration,
            "watch_ratio": f"{watch_ratio:.6f}",
        }
    )
    user_seen[user_id].add(item_id)


if __name__ == "__main__":
    main()
