from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.ranking.negative_sampling import (
    DEFAULT_NEGATIVE_MIX,
    sample_exposure_aware_negatives,
    stats_to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exposure-aware ranker train samples from temporal candidate rows."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_dir", default=None, help="Directory containing train/val/test_rank_samples.parquet.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--report_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument(
        "--strategy",
        default="mixed_negative",
        choices=[
            "random_negative",
            "exposed_low_watch_negative",
            "retrieved_hard_negative",
            "popularity_hard_negative",
            "mixed",
            "mixed_negative",
        ],
    )
    parser.add_argument("--negative_mix", default=None, help="Comma-separated type=weight values for mixed strategy.")
    parser.add_argument("--negatives_per_positive", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_watch_threshold", type=float, default=0.3)
    parser.add_argument("--popularity_quantile", type=float, default=0.8)
    parser.add_argument("--future_positive_splits", default="")
    parser.add_argument("--include_injected_positives", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir or f"data/features/{args.dataset}")
    output_dir = Path(args.output_dir or input_dir)
    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    experiment_name = args.experiment_name or output_dir.name
    report_dir = Path(args.report_dir or f"reports/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train_rank_samples.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing temporal train ranker samples: {train_path}")
    train = pd.read_parquet(train_path).reset_index(drop=True)
    interactions_path = processed_dir / "interactions.parquet"
    interactions = pd.read_parquet(interactions_path) if interactions_path.exists() else None
    temporal_meta = _load_meta(input_dir / "ranker_candidate_meta.json").get("temporal_train", {})
    cutoff = temporal_meta.get("cutoff_timestamp")
    if interactions is not None:
        if cutoff is None:
            raise ValueError("Exposure negatives require temporal prefix cutoff metadata.")
        interactions = interactions[(interactions["split"] == "train") & (interactions.timestamp <= cutoff)]
    mix = _parse_mix(args.negative_mix)
    future_splits = [value.strip() for value in args.future_positive_splits.split(",") if value.strip()]

    sampled, stats = sample_exposure_aware_negatives(
        train,
        interactions=interactions,
        strategy=args.strategy,
        mix=mix,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        include_injected_positives=args.include_injected_positives,
        low_watch_threshold=args.low_watch_threshold,
        popularity_quantile=args.popularity_quantile,
        future_positive_splits=future_splits,
    )
    sampled.to_parquet(output_dir / "train_rank_samples.parquet", index=False)
    _copy_optional(input_dir, output_dir, ["val_rank_samples.parquet", "test_rank_samples.parquet", "encoders.pkl"])
    meta = _load_meta(input_dir / "ranker_candidate_meta.json")
    meta["negative_sampling"] = {
        "strategy": stats.get("strategy"),
        "negative_mix": stats.get("negative_mix", {}),
        "negatives_per_positive": args.negatives_per_positive,
        "low_watch_threshold": args.low_watch_threshold,
        "popularity_quantile": args.popularity_quantile,
        "future_positive_splits": future_splits,
        "stats": stats,
    }
    splits = meta.setdefault("splits", {})
    splits["train"] = _split_stats(sampled)
    (output_dir / "ranker_candidate_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path = report_dir / "negative_sampling_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "negative_sampling_stats.md").write_text(stats_to_markdown(stats), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "stats": stats}, ensure_ascii=False, indent=2))


def _parse_mix(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_NEGATIVE_MIX)
    result: dict[str, float] = {}
    for part in [value.strip() for value in text.split(",") if value.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid negative mix component: {part}")
        name, value = part.split("=", 1)
        result[name.strip()] = float(value)
    return result


def _copy_optional(input_dir: Path, output_dir: Path, filenames: list[str]) -> None:
    for filename in filenames:
        source = input_dir / filename
        target = output_dir / filename
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "positives": 0,
            "retrieved_positives": 0,
            "injected_positives": 0,
            "positive_injection_rate": 0.0,
            "candidate_positive_recall": 0.0,
            "negatives": 0,
            "users": 0,
        }
    labels = pd.to_numeric(frame["label"], errors="coerce").fillna(0)
    positives = int((labels > 0).sum())
    retrieved_values = (
        frame["is_retrieved_positive"] if "is_retrieved_positive" in frame.columns else pd.Series(0, index=frame.index)
    )
    injected_values = (
        frame["is_injected_positive"] if "is_injected_positive" in frame.columns else pd.Series(0, index=frame.index)
    )
    retrieved = int(pd.to_numeric(retrieved_values, errors="coerce").fillna(0).sum())
    injected = int(pd.to_numeric(injected_values, errors="coerce").fillna(0).sum())
    return {
        "rows": int(len(frame)),
        "positives": positives,
        "retrieved_positives": retrieved,
        "injected_positives": injected,
        "positive_injection_rate": float(injected / positives) if positives else 0.0,
        "candidate_positive_recall": float(retrieved / positives) if positives else 0.0,
        "negatives": int((labels <= 0).sum()),
        "users": int(frame["user_id"].nunique()) if "user_id" in frame.columns else 0,
    }


if __name__ == "__main__":
    main()
