from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment artifacts for resume/interview-safe claims.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--reports_dir", default=None)
    parser.add_argument("--require_real", action="store_true", help="Fail if meta marks sample fallback or simulated data.")
    parser.add_argument("--min_candidate_recall", type=float, default=None)
    parser.add_argument("--max_positive_injection_rate", type=float, default=None)
    parser.add_argument("--min_recall100", type=float, default=None)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    reports_dir = Path(args.reports_dir or f"reports/{args.dataset}")

    failures: list[str] = []
    warnings: list[str] = []
    meta = _read_json(processed_dir / "meta.json")
    candidate_meta = _read_json(features_dir / "ranker_candidate_meta.json")
    metrics = _read_metrics(reports_dir / "metrics_summary.csv")

    if args.require_real:
        if meta.get("is_sample_fallback"):
            failures.append("Processed dataset is sample fallback.")
        if meta.get("is_simulated"):
            failures.append("Processed dataset is simulated.")
        if not meta:
            failures.append("Missing processed meta.json.")

    test_stats = _candidate_stats(candidate_meta, "test")
    if args.min_candidate_recall is not None:
        value = float(test_stats.get("candidate_positive_recall", 0.0))
        if value < args.min_candidate_recall:
            failures.append(
                f"candidate_positive_recall={value:.6f} < {args.min_candidate_recall:.6f}"
            )
    if args.max_positive_injection_rate is not None:
        value = float(test_stats.get("positive_injection_rate", 1.0))
        if value > args.max_positive_injection_rate:
            failures.append(
                f"positive_injection_rate={value:.6f} > {args.max_positive_injection_rate:.6f}"
            )

    best_recall100 = _best_metric(metrics, "Recall@100")
    if args.min_recall100 is not None and best_recall100 < args.min_recall100:
        failures.append(f"best Recall@100={best_recall100:.6f} < {args.min_recall100:.6f}")
    if metrics.empty:
        warnings.append("Missing or empty metrics_summary.csv.")
    elif "status" in metrics.columns:
        failed_rows = metrics[~metrics["status"].astype(str).str.startswith("ok")]
        if not failed_rows.empty:
            warnings.append(f"{len(failed_rows)} matrix rows are not ok.")

    result = {
        "dataset": args.dataset,
        "meta": {
            "num_users": meta.get("num_users"),
            "num_items": meta.get("num_items"),
            "num_interactions": meta.get("num_interactions"),
            "is_sample_fallback": meta.get("is_sample_fallback"),
            "is_simulated": meta.get("is_simulated"),
            "data_origin": meta.get("data_origin"),
        },
        "test_candidate_positive_recall": test_stats.get("candidate_positive_recall"),
        "test_positive_injection_rate": test_stats.get("positive_injection_rate"),
        "best_recall100": best_recall100,
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _candidate_stats(candidate_meta: dict[str, Any], split: str) -> dict[str, Any]:
    splits = candidate_meta.get("splits", {}) if isinstance(candidate_meta, dict) else {}
    if isinstance(splits, dict) and split in splits:
        return dict(splits[split])
    overall = candidate_meta.get("overall", {}) if isinstance(candidate_meta, dict) else {}
    return dict(overall) if isinstance(overall, dict) else {}


def _best_metric(metrics: pd.DataFrame, column: str) -> float:
    if metrics.empty or column not in metrics.columns:
        return 0.0
    values = pd.to_numeric(metrics[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else 0.0


if __name__ == "__main__":
    main()
