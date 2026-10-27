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

from llm4rec.evaluation.metrics import ranking_metrics
from llm4rec.retrieval.trend import TrendRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune TrendRetriever on validation split and optionally save best.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--output_dir", default=None, help="Retriever output directory.")
    parser.add_argument("--reports_dir", default=None)
    parser.add_argument("--half_life_days", default="0.25,0.5,1,2,3,5,7,14")
    parser.add_argument("--category_weights", default="0,0.2,0.4")
    parser.add_argument("--tag_weights", default="0,0.1,0.2")
    parser.add_argument("--popularity_floors", default="0.0,0.01,0.02")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument("--save_best", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    output_dir = Path(args.output_dir or f"models/{args.dataset}/retrievers")
    reports_dir = Path(args.reports_dir or f"reports/{args.dataset}_trend_tuning")
    reports_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(processed_dir / "items.parquet")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    train = interactions[interactions["split"] == "train"].reset_index(drop=True)
    val = interactions[interactions["split"] == "val"].reset_index(drop=True)
    test = interactions[interactions["split"] == "test"].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    best_model: TrendRetriever | None = None
    best_score = -1.0
    best_params: dict[str, float] = {}
    half_life_values = _floats(args.half_life_days)
    category_weight_values = _floats(args.category_weights)
    tag_weight_values = _floats(args.tag_weights)
    popularity_floor_values = _floats(args.popularity_floors)
    total_runs = (
        len(half_life_values)
        * len(category_weight_values)
        * len(tag_weight_values)
        * len(popularity_floor_values)
    )
    run_idx = 0
    for half_life in half_life_values:
        for category_weight in category_weight_values:
            for tag_weight in tag_weight_values:
                for popularity_floor in popularity_floor_values:
                    run_idx += 1
                    params = {
                        "half_life_days": half_life,
                        "category_weight": category_weight,
                        "tag_weight": tag_weight,
                        "popularity_floor": popularity_floor,
                    }
                    model = TrendRetriever(**params).fit(train, items)
                    val_metrics = _evaluate_split(model, val, seen_splits=[train], top_k=args.top_k)
                    row = {**params, **{f"val_{key}": value for key, value in val_metrics.items()}}
                    rows.append(row)
                    score = float(val_metrics.get("Recall@100", 0.0))
                    improved = score > best_score
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = params
                    if _should_log_progress(run_idx, total_runs, args.progress_every, improved):
                        print(
                            "[trend-tune] "
                            f"{run_idx}/{total_runs} "
                            f"params={params} "
                            f"val_Recall@100={score:.6f} "
                            f"best={best_score:.6f}",
                            file=sys.stderr,
                            flush=True,
                        )

    results = pd.DataFrame(rows).sort_values("val_Recall@100", ascending=False)
    results.to_csv(reports_dir / "trend_tuning_results.csv", index=False)

    if best_model is None:
        raise RuntimeError("No trend tuning candidates were evaluated.")
    test_metrics = _evaluate_split(best_model, test, seen_splits=[train, val], top_k=args.top_k)
    summary = {
        "dataset": args.dataset,
        "selection_metric": "val_Recall@100",
        "best_params": best_params,
        "best_val": {
            key.replace("val_", ""): _json_number(value)
            for key, value in results.iloc[0].items()
            if str(key).startswith("val_")
        },
        "test_metrics_for_best": test_metrics,
        "evaluated_runs": int(len(results)),
    }
    (reports_dir / "trend_tuning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "trend_tuning_report.md").write_text(
        _render_report(summary, results.head(20)),
        encoding="utf-8",
    )
    if args.save_best:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model.save(output_dir / "trend")
        (output_dir / "trend" / "metrics.json").write_text(
            json.dumps(
                {
                    **{f"val_{key}": value for key, value in summary["best_val"].items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    "selected_by": "val_Recall@100",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _evaluate_split(
    retriever: TrendRetriever,
    target: pd.DataFrame,
    seen_splits: list[pd.DataFrame],
    top_k: int,
) -> dict[str, float]:
    positives = target[target["label"] == 1]
    if positives.empty:
        return {"users": 0.0}
    relevant_by_user = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in positives.groupby("user_id")
    }
    seen = _seen_by_user(seen_splits)
    per_user = []
    candidate_hits = 0
    positives_count = 0
    for user_id, relevant in relevant_by_user.items():
        expanded_top_k = top_k + len(seen.get(user_id, set()))
        recommendations = [
            candidate.item_id
            for candidate in retriever.retrieve(user_id, top_k=expanded_top_k)
            if candidate.item_id not in seen.get(user_id, set())
        ][:top_k]
        per_user.append(ranking_metrics(recommendations, relevant, [10, 50, 100, 200]))
        candidate_hits += len(set(recommendations) & relevant)
        positives_count += len(relevant)
    metrics = _mean_metrics(per_user)
    metrics["candidate_positive_recall"] = float(candidate_hits / positives_count) if positives_count else 0.0
    metrics["users"] = float(len(relevant_by_user))
    return metrics


def _seen_by_user(frames: list[pd.DataFrame]) -> dict[str, set[str]]:
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    positives = combined[combined["label"] == 1]
    return {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in positives.groupby("user_id")
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


def _floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _json_number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric


def _should_log_progress(run_idx: int, total_runs: int, progress_every: int, improved: bool) -> bool:
    if improved or run_idx == 1 or run_idx == total_runs:
        return True
    return progress_every > 0 and run_idx % progress_every == 0


def _render_report(summary: dict[str, Any], top_results: pd.DataFrame) -> str:
    return "\n\n".join(
        [
            "# Trend Retriever Tuning",
            f"Dataset: `{summary['dataset']}`",
            f"Selected by: `{summary['selection_metric']}`",
            "## Best Params\n\n```json\n"
            + json.dumps(summary["best_params"], ensure_ascii=False, indent=2)
            + "\n```",
            "## Test Metrics For Best\n\n```json\n"
            + json.dumps(summary["test_metrics_for_best"], ensure_ascii=False, indent=2)
            + "\n```",
            "## Top Validation Runs\n\n" + _df_to_markdown(top_results),
        ]
    ) + "\n"


def _df_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_format_cell(row[column]) for column in frame.columns) + " |")
    return "\n".join([header, sep, *rows])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
