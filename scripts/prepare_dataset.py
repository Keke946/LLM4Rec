from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4rec.datasets import DATASET_ADAPTERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a public recommendation dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_ADAPTERS), help="Dataset name.")
    parser.add_argument("--raw_dir", required=True, help="Raw dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Processed output directory.")
    parser.add_argument(
        "--positive_watch_ratio_threshold",
        type=float,
        default=1.0,
        help="KuaiRec positive label threshold.",
    )
    parser.add_argument(
        "--positive_rating_threshold",
        type=float,
        default=4.0,
        help="MovieLens positive rating threshold.",
    )
    parser.add_argument(
        "--kuairec_matrix",
        choices=["auto", "small", "big"],
        default="auto",
        help="KuaiRec interaction matrix to load. auto preserves adapter defaults.",
    )
    parser.add_argument("--sample_users", type=int, default=None, help="Keep the most active N users.")
    parser.add_argument("--sample_items", type=int, default=None, help="Keep the most frequent N items after user sampling.")
    parser.add_argument("--max_interactions", type=int, default=None, help="Keep at most N interactions after sampling.")
    parser.add_argument(
        "--max_interactions_strategy",
        choices=["temporal", "random"],
        default="temporal",
        help="How to choose interactions when --max_interactions is set.",
    )
    parser.add_argument(
        "--max_interactions_seed",
        type=int,
        default=42,
        help="Random seed used when --max_interactions_strategy=random.",
    )
    parser.add_argument(
        "--no_sample_fallback",
        action="store_true",
        help="Fail if raw_dir is missing instead of using examples/sample_data.",
    )
    parser.add_argument(
        "--allow_simulated_raw",
        action="store_true",
        help="Allow raw directories marked by simulated_data_manifest.json.",
    )
    parser.add_argument("--min_users", type=int, default=None, help="Fail if processed user count is lower.")
    parser.add_argument("--min_items", type=int, default=None, help="Fail if processed item count is lower.")
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=None,
        help="Fail if processed interaction count is lower.",
    )
    parser.add_argument("--split_strategy", choices=["per_user_temporal", "global_temporal"],
                        default="per_user_temporal")
    parser.add_argument("--allow_sample_fallback", action="store_true",
                        help="Explicitly permit tiny example fallback for smoke tests only.")
    args = parser.parse_args()

    requested_raw_dir = Path(args.raw_dir)
    raw_dir = requested_raw_dir
    is_sample_fallback = False
    data_origin = str(raw_dir)
    sample_dir = PROJECT_ROOT / "examples" / "sample_data" / args.dataset
    if not raw_dir.exists() and sample_dir.exists():
        if args.no_sample_fallback or not args.allow_sample_fallback:
            raise FileNotFoundError(
                f"Raw dir {raw_dir} not found and --no_sample_fallback was set."
            )
        print(f"Raw dir {raw_dir} not found. Falling back to sample data at {sample_dir}.")
        raw_dir = sample_dir
        is_sample_fallback = True
        data_origin = str(sample_dir)
    simulation_manifest = raw_dir / "simulated_data_manifest.json"
    is_simulated = simulation_manifest.exists()
    if is_simulated and args.no_sample_fallback and not args.allow_simulated_raw:
        raise RuntimeError(
            f"Raw dir {raw_dir} is simulated data ({simulation_manifest}). "
            "Official real-data runs must use real raw data, or pass --allow_simulated_raw "
            "for a clearly marked simulated smoke run."
        )

    adapter_cls = DATASET_ADAPTERS[args.dataset]
    adapter = adapter_cls(
        raw_dir=raw_dir,
        positive_watch_ratio_threshold=args.positive_watch_ratio_threshold,
        positive_rating_threshold=args.positive_rating_threshold,
        kuairec_matrix=args.kuairec_matrix,
        split_strategy=args.split_strategy,
        sample_users=args.sample_users,
        sample_items=args.sample_items,
        max_interactions=args.max_interactions,
        max_interactions_strategy=args.max_interactions_strategy,
        max_interactions_seed=args.max_interactions_seed,
        is_sample_fallback=is_sample_fallback,
        data_origin=data_origin,
        requested_raw_dir=str(requested_raw_dir),
        is_simulated=is_simulated,
        simulation_manifest=str(simulation_manifest) if is_simulated else None,
    )
    meta = adapter.save_processed(args.output_dir)
    _validate_minimums(meta, args)
    print(f"Prepared {args.dataset} at {args.output_dir}")
    print(meta)


def _validate_minimums(meta, args) -> None:
    failures = []
    if args.min_users is not None and meta.num_users < args.min_users:
        failures.append(f"num_users={meta.num_users} < min_users={args.min_users}")
    if args.min_items is not None and meta.num_items < args.min_items:
        failures.append(f"num_items={meta.num_items} < min_items={args.min_items}")
    if args.min_interactions is not None and meta.num_interactions < args.min_interactions:
        failures.append(
            f"num_interactions={meta.num_interactions} < min_interactions={args.min_interactions}"
        )
    if failures:
        raise RuntimeError("Processed dataset failed minimum-size checks: " + "; ".join(failures))


if __name__ == "__main__":
    main()
