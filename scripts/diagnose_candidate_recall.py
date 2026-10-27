from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose candidate-pool recall by source and popularity buckets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    features_dir = Path(args.features_dir or f"data/features/{args.dataset}")
    processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
    output_dir = Path(args.output_dir or f"reports/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = features_dir / f"{args.split}_rank_samples.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Ranker candidate sample file not found: {samples_path}")

    samples = pd.read_parquet(samples_path)
    interactions = _read_optional_parquet(processed_dir / "interactions.parquet")
    summary, popularity_table, history_table, source_table = diagnose(samples, interactions, args.split)

    (output_dir / "candidate_recall_diagnostics.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "summary": summary,
                "by_item_popularity": popularity_table.to_dict(orient="records"),
                "by_history_len": history_table.to_dict(orient="records"),
                "by_source": source_table.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    popularity_table.to_csv(output_dir / "candidate_recall_by_item_popularity.csv", index=False)
    history_table.to_csv(output_dir / "candidate_recall_by_history_len.csv", index=False)
    source_table.to_csv(output_dir / "candidate_recall_by_source.csv", index=False)
    (output_dir / "candidate_recall_diagnostics.md").write_text(
        _render_markdown(args.dataset, args.split, summary, popularity_table, history_table, source_table),
        encoding="utf-8",
    )
    print(f"Wrote candidate recall diagnostics to {output_dir}")


def diagnose(
    samples: pd.DataFrame,
    interactions: pd.DataFrame | None,
    split: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = samples.copy()
    positives = frame[frame["label"] == 1].copy()
    positive_count = int(len(positives))
    retrieved_count = int(positives.get("is_retrieved_positive", pd.Series(dtype=int)).sum())
    injected_count = int(positives.get("is_injected_positive", pd.Series(dtype=int)).sum())
    summary = {
        "rows": int(len(frame)),
        "positives": positive_count,
        "retrieved_positives": retrieved_count,
        "injected_positives": injected_count,
        "candidate_positive_recall": float(retrieved_count / positive_count) if positive_count else 0.0,
        "positive_injection_rate": float(injected_count / positive_count) if positive_count else 0.0,
    }
    if interactions is not None and not interactions.empty:
        summary.update(_raw_positive_overlap(interactions, split))

    popularity_table = _bucket_table(
        positives,
        column="item_popularity",
        bins=[-1, 0, 1, 3, 10, 30, 1000000000],
        labels=["cold_0", "pop_1", "pop_2_3", "pop_4_10", "pop_11_30", "pop_31_plus"],
        bucket_column="item_popularity_bucket",
    )
    history_table = _bucket_table(
        positives,
        column="history_len",
        bins=[-1, 0, 5, 20, 50, 1000000000],
        labels=["h0", "h1_5", "h6_20", "h21_50", "h50_plus"],
        bucket_column="history_len_bucket",
    )
    source_table = _source_table(frame, positives, positive_count)
    return summary, popularity_table, history_table, source_table


def _bucket_table(
    positives: pd.DataFrame,
    column: str,
    bins: list[float],
    labels: list[str],
    bucket_column: str,
) -> pd.DataFrame:
    if positives.empty or column not in positives.columns:
        return pd.DataFrame(columns=[bucket_column, "positives", "retrieved_positives", "candidate_positive_recall"])
    work = positives.copy()
    work["_retrieved"] = work["is_retrieved_positive"].astype(bool)
    work[bucket_column] = pd.cut(work[column].fillna(0), bins=bins, labels=labels)
    grouped = work.groupby(bucket_column, observed=False)["_retrieved"].agg(["count", "sum", "mean"]).reset_index()
    grouped = grouped.rename(
        columns={
            "count": "positives",
            "sum": "retrieved_positives",
            "mean": "candidate_positive_recall",
        }
    )
    grouped["retrieved_positives"] = grouped["retrieved_positives"].astype(int)
    return grouped


def _source_table(frame: pd.DataFrame, positives: pd.DataFrame, positive_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = sorted(_all_sources(frame))
    for source in sources:
        candidate_rows = int(frame["retrieval_sources"].apply(lambda value: source in _as_sources(value)).sum())
        retrieved_hits = int(
            positives[
                positives["is_retrieved_positive"].astype(bool)
                & positives["retrieval_sources"].apply(lambda value: source in _as_sources(value))
            ].shape[0]
        )
        rows.append(
            {
                "source": source,
                "candidate_rows": candidate_rows,
                "retrieved_positive_hits": retrieved_hits,
                "credited_positive_recall": float(retrieved_hits / positive_count) if positive_count else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["source", "candidate_rows", "retrieved_positive_hits", "credited_positive_recall"]).sort_values("retrieved_positive_hits", ascending=False).reset_index(drop=True)


def _all_sources(frame: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    if "retrieval_sources" not in frame.columns:
        return result
    for value in frame["retrieval_sources"]:
        result.update(_as_sources(value))
    return {source for source in result if source}


def _as_sources(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(part) for part in parsed if str(part)]
            except (SyntaxError, ValueError):
                pass
        return [part for part in text.split("+") if part]
    if isinstance(value, (list, tuple, set)):
        return [str(part) for part in value if str(part)]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(part) for part in converted if str(part)]
    return []


def _raw_positive_overlap(interactions: pd.DataFrame, split: str) -> dict[str, Any]:
    positives = interactions[interactions["label"] == 1].copy()
    train_items = set(positives[positives["split"] == "train"]["item_id"].astype(str))
    split_positive_items = positives[positives["split"] == split]["item_id"].astype(str)
    if split_positive_items.empty:
        return {"split_positive_item_in_train_ratio": 0.0}
    return {
        "split_positive_item_in_train_ratio": float(split_positive_items.isin(train_items).mean()),
        "split_unique_positive_items": int(split_positive_items.nunique()),
    }


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _render_markdown(
    dataset: str,
    split: str,
    summary: dict[str, Any],
    popularity_table: pd.DataFrame,
    history_table: pd.DataFrame,
    source_table: pd.DataFrame,
) -> str:
    return "\n\n".join(
        [
            f"# Candidate Recall Diagnostics: {dataset}",
            f"Split: `{split}`",
            "## Summary\n\n" + _dict_table(summary),
            "## By Item Popularity\n\n" + _markdown_table(popularity_table),
            "## By History Length\n\n" + _markdown_table(history_table),
            "## By Retrieval Source\n\n" + _markdown_table(source_table),
            "## Reading Guide\n\n"
            "`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. "
            "High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. "
            "Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.",
        ]
    ) + "\n"


def _dict_table(values: dict[str, Any]) -> str:
    rows = [{"metric": key, "value": value} for key, value in values.items()]
    return _markdown_table(pd.DataFrame(rows))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].apply(_format_value)
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
