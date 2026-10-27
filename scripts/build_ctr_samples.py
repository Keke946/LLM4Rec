from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.config import DEFAULT_PATHS, ensure_project_dirs
from llm4rec.data import build_tag_binarizer, load_user_behavior, load_videos
from llm4rec.models import binarize_tags


def build_ctr_samples(negative_ratio: int, output: str, seed: int = 42) -> pd.DataFrame:
    ensure_project_dirs()
    rng = random.Random(seed)
    videos = load_videos()
    behavior = load_user_behavior()
    binarizer = build_tag_binarizer(videos)
    video_by_id = videos.set_index("video_id")
    all_video_ids = set(videos["video_id"])

    rows: list[list[int]] = []
    for _, info in behavior.items():
        user_tags = info["preferred_tags"]
        watched = [video_id for video_id in info["watched_videos"] if video_id in all_video_ids]
        negative_pool = sorted(all_video_ids - set(watched))
        negatives = rng.sample(
            negative_pool,
            k=min(len(negative_pool), max(1, len(watched) * negative_ratio)),
        )

        user_vec = binarize_tags(user_tags, binarizer)
        for video_id, label in [(video_id, 1) for video_id in watched] + [
            (video_id, 0) for video_id in negatives
        ]:
            video_vec = binarize_tags(video_by_id.loc[video_id, "tags"], binarizer)
            rows.append(user_vec + video_vec + [label])

    columns = (
        [f"u_{tag}" for tag in binarizer.classes_]
        + [f"v_{tag}" for tag in binarizer.classes_]
        + ["label"]
    )
    samples = pd.DataFrame(rows, columns=columns)
    samples.to_csv(output, index=False)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CTR training samples.")
    parser.add_argument("--negative-ratio", type=int, default=3, help="Negative samples per positive.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default=str(DEFAULT_PATHS.ctr_samples_csv), help="Output CSV path.")
    args = parser.parse_args()
    samples = build_ctr_samples(args.negative_ratio, args.output, args.seed)
    positive_rate = samples["label"].mean() if len(samples) else 0.0
    print(f"Wrote {len(samples)} samples to {args.output} (positive_rate={positive_rate:.3f})")


if __name__ == "__main__":
    main()
