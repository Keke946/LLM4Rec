from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4rec.features import build_ranking_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ranking features from processed parquet files.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--processed_dir", required=True, help="Processed dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Feature output directory.")
    parser.add_argument("--max_history_len", type=int, default=20, help="Maximum history sequence length.")
    args = parser.parse_args()

    encoders = build_ranking_features(args.processed_dir, args.output_dir, args.max_history_len)
    print(
        f"Built features for {args.dataset}: "
        f"users={len(encoders.user)} items={len(encoders.item)} "
        f"categories={len(encoders.category)} tags={len(encoders.tag)}"
    )


if __name__ == "__main__":
    main()
