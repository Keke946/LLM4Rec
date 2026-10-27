from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.evaluation.evaluator import OfflineEvaluator, save_evaluation_outputs
from llm4rec.explainer import reject_hallucinated_item_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline recommendation experiment matrix.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config path.")
    parser.add_argument("--processed_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--retriever_dir", default=None, help="Retriever artifact directory.")
    parser.add_argument("--features_dir", default=None, help="Feature directory.")
    parser.add_argument("--ranker_root", default=None, help="Ranker root directory.")
    parser.add_argument("--output_dir", default=None, help="Report output directory.")
    parser.add_argument(
        "--retrievers",
        default=None,
        help="Comma-separated retrievers. Use fusion for all configured retrievers.",
    )
    parser.add_argument("--rankers", default=None, help="Comma-separated rankers; use none for retrieval-only.")
    parser.add_argument("--rerank_modes", default=None, help="Comma-separated rerank modes.")
    parser.add_argument("--rerankers", default=None, help="Comma-separated business rerankers.")
    parser.add_argument("--candidate_pool_k", type=int, default=None)
    parser.add_argument("--cutoffs", default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--filter_seen_splits", default=None)
    parser.add_argument("--retrieval_weight", type=float, default=None)
    parser.add_argument("--ranker_weight", type=float, default=None)
    parser.add_argument(
        "--source_weights",
        default=None,
        help="Optional comma-separated RRF source weights, e.g. sasrec=1.2,metadata=0.4.",
    )
    parser.add_argument("--skip_auto_train_rankers", action="store_true", help="Do not train missing ranker artifacts.")
    parser.add_argument("--max_runs", type=int, default=None, help="Optional cap for quick smoke runs.")
    args = parser.parse_args()

    config = _load_config(args.config)
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    evaluation = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    retrieval_cfg = config.get("retrieval", {}) if isinstance(config.get("retrieval", {}), dict) else {}
    ranking_cfg = config.get("ranking", {}) if isinstance(config.get("ranking", {}), dict) else {}
    features_cfg = config.get("features", {}) if isinstance(config.get("features", {}), dict) else {}

    processed_dir = Path(args.processed_dir or paths.get("processed_dir") or f"data/processed/{args.dataset}")
    retriever_dir = args.retriever_dir or paths.get("retriever_dir") or f"models/{args.dataset}/retrievers"
    features_dir = args.features_dir or paths.get("features_dir") or f"data/features/{args.dataset}"
    ranker_root = args.ranker_root or paths.get("ranker_root") or f"models/{args.dataset}"
    output_dir = Path(args.output_dir or f"reports/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    configured_retrievers = _as_list(retrieval_cfg.get("retrievers")) or [
        "popular",
        "itemcf",
        "bpr",
        "lightgcn",
        "two_tower",
        "sasrec",
        "content_faiss",
        "metadata",
    ]
    retrievers = _split(args.retrievers) if args.retrievers else configured_retrievers + ["fusion"]
    rankers = _split(args.rankers) if args.rankers else (_as_list(ranking_cfg.get("rankers")) or ["none", "embedding_mlp", "deepfm", "din"])
    if "none" not in rankers:
        rankers = ["none"] + rankers
    rerank_modes = _split(args.rerank_modes) if args.rerank_modes else ["retrieval_only", "ranker_only", "retrieval_ranker_fusion"]
    rerankers = _split(args.rerankers) if args.rerankers else _as_list(evaluation.get("rerankers")) or ["none"]
    cutoffs = (
        [int(value) for value in _split(args.cutoffs)]
        if args.cutoffs
        else [int(value) for value in _as_list(evaluation.get("cutoffs"))] or [5, 10, 20, 50, 100]
    )
    filter_seen_splits = (
        _split(args.filter_seen_splits)
        if args.filter_seen_splits
        else [str(value) for value in _as_list(evaluation.get("filter_seen_splits"))] or ["train", "val"]
    )
    candidate_pool_k = int(args.candidate_pool_k or evaluation.get("candidate_pool_k") or 500)
    final_top_k = int(args.final_top_k or evaluation.get("final_top_k") or 10)
    retrieval_weight = float(args.retrieval_weight or evaluation.get("retrieval_weight") or 0.7)
    ranker_weight = float(args.ranker_weight or evaluation.get("ranker_weight") or 0.3)
    source_weights = _parse_source_weights(args.source_weights)
    if not source_weights:
        source_weights = {
            str(name): float(value)
            for name, value in (retrieval_cfg.get("source_weights") or {}).items()
        }
    candidate_meta = _load_candidate_meta(Path(features_dir))
    llm_safety = _llm_safety_eval()

    if not args.skip_auto_train_rankers:
        _ensure_rankers(
            args.dataset,
            rankers,
            Path(features_dir),
            Path(ranker_root),
            ranking_cfg,
            features_cfg,
        )

    evaluator = OfflineEvaluator(
        processed_dir=processed_dir,
        retriever_dir=retriever_dir,
        features_dir=features_dir,
        ranker_root=ranker_root,
    )
    retriever_sets = _retriever_sets(retrievers, configured_retrievers)
    rows: list[dict[str, object]] = []
    failure_cases: dict[str, object] = {}
    first_success: dict[str, object] | None = None
    run_count = 0

    for display_name, retriever_set in retriever_sets:
        for ranker in rankers:
            modes = ["retrieval_only"] if ranker == "none" else rerank_modes
            for mode in modes:
                for business_reranker in rerankers:
                    run_count += 1
                    if args.max_runs is not None and run_count > args.max_runs:
                        break
                    combo_id = (
                        f"retrievers={display_name}|ranker={ranker}|"
                        f"mode={mode}|reranker={business_reranker}"
                    )
                    try:
                        result = evaluator.evaluate_retrievers(
                            retriever_set,
                            ranker_name=None if ranker == "none" else ranker,
                            candidate_pool_k=candidate_pool_k,
                            cutoffs=cutoffs,
                            final_top_k=final_top_k,
                            filter_seen_splits=filter_seen_splits,
                            rerank_mode=mode,
                            retrieval_weight=retrieval_weight,
                            ranker_weight=ranker_weight,
                            reranker=business_reranker,
                            source_weights=source_weights,
                        )
                        if first_success is None:
                            first_success = result
                        row = _matrix_row(
                            args.dataset,
                            display_name,
                            ranker,
                            mode,
                            business_reranker,
                            result,
                            ranker_root,
                            candidate_meta,
                            source_weights,
                        )
                        rows.append(row)
                        failure_cases[combo_id] = result.get("failure_cases", [])
                    except Exception as exc:
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "retrievers": display_name,
                                "fusion": "RRF" if len(retriever_set) > 1 else "single",
                                "ranker": ranker,
                                "rerank_mode": mode,
                                "business_reranker": business_reranker,
                                "candidate_pool_k": candidate_pool_k,
                                "final_top_k": final_top_k,
                                "source_weights": json.dumps(source_weights, ensure_ascii=False, sort_keys=True),
                                "status": f"failed: {type(exc).__name__}: {exc}",
                            }
                        )
                        failure_cases[combo_id] = [{"error": repr(exc)}]
                if args.max_runs is not None and run_count >= args.max_runs:
                    break
            if args.max_runs is not None and run_count >= args.max_runs:
                break
        if args.max_runs is not None and run_count >= args.max_runs:
            break

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "failure_cases.json").write_text(
        json.dumps(failure_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_matrix_report(
        metrics,
        processed_dir,
        output_dir / "experiment_matrix.md",
        args.dataset,
        filter_seen_splits,
        candidate_pool_k,
        cutoffs,
        configured_retrievers,
        rankers,
        llm_safety,
        candidate_meta,
        source_weights,
    )
    if first_success is not None:
        meta_path = processed_dir / "meta.json"
        if meta_path.exists():
            first_success["data_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        save_evaluation_outputs(first_success, output_dir, args.dataset)
        metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
        (output_dir / "failure_cases.json").write_text(
            json.dumps(failure_cases, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_formal_report(
            metrics,
            processed_dir,
            output_dir / "experiment_report.md",
            args.dataset,
            filter_seen_splits,
            candidate_pool_k,
            cutoffs,
            configured_retrievers,
            rankers,
            candidate_meta,
            failure_cases,
            llm_safety,
            source_weights,
        )
    print(f"Saved experiment matrix to {output_dir}")


def _matrix_row(
    dataset: str,
    retriever_display: str,
    ranker: str,
    mode: str,
    business_reranker: str,
    result: dict[str, object],
    ranker_root: str | Path,
    candidate_meta: dict[str, Any],
    source_weights: dict[str, float],
) -> dict[str, object]:
    summary = dict(result["summary"])
    summary["Coverage"] = summary.get("CatalogCoverage")
    before = result.get("business_rerank_before_summary") or {}
    after = result.get("business_rerank_summary") or {}
    ranker_metrics = _ranker_metrics(Path(ranker_root), ranker)
    candidate_stats = _candidate_stats(candidate_meta, "test")
    return {
        "dataset": dataset,
        "retrievers": retriever_display,
        "fusion": "RRF" if "," in retriever_display or retriever_display == "fusion" else "single",
        "ranker": ranker,
        "rerank_mode": mode,
        "business_reranker": business_reranker,
        "candidate_pool_k": result["candidate_pool_k"],
        "final_top_k": result["final_top_k"],
        "source_weights": json.dumps(source_weights, ensure_ascii=False, sort_keys=True),
        "candidate_positive_recall": candidate_stats.get("candidate_positive_recall"),
        "positive_injection_rate": candidate_stats.get("positive_injection_rate"),
        "status": "ok" if mode == "retrieval_only" or result.get("ranker_available") else "ranker_missing_or_skipped",
        "AUC": ranker_metrics.get("AUC"),
        "LogLoss": ranker_metrics.get("LogLoss"),
        "NDCG@10_before_business_rerank": before.get("NDCG@10"),
        "NDCG@10_after_business_rerank": after.get("NDCG@10"),
        "Coverage_before_business_rerank": before.get("CatalogCoverage"),
        "Coverage_after_business_rerank": after.get("CatalogCoverage"),
        "Diversity_before_business_rerank": before.get("Diversity"),
        "Diversity_after_business_rerank": after.get("Diversity"),
        "TailRatio_before_business_rerank": before.get("TailRatio"),
        "TailRatio_after_business_rerank": after.get("TailRatio"),
        **summary,
    }


def _write_matrix_report(
    metrics: pd.DataFrame,
    processed_dir: Path,
    report_path: Path,
    dataset: str,
    filter_seen_splits: list[str],
    candidate_pool_k: int,
    cutoffs: list[int],
    retrievers: list[str],
    rankers: list[str],
    llm_safety: dict[str, float],
    candidate_meta: dict[str, Any],
    source_weights: dict[str, float],
) -> None:
    meta_path = processed_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ok = metrics[metrics["status"].astype(str).str.startswith("ok")] if "status" in metrics else metrics
    top_rows = _sort_metric(ok, "Recall@10").head(12)
    text = f"""# LLM4Rec Experiment Matrix

Dataset: `{dataset}` ({'subset' if meta.get('is_subset') else 'full or unmarked'})

## Dataset Statistics

```json
{json.dumps(meta, ensure_ascii=False, indent=2)}
```

## Experiment Setup

- Positive label definition: KuaiRec watch ratio threshold, MovieLens rating threshold, or MIND clicked impression labels depending on adapter.
- Split strategy: per-user temporal split; MIND additionally keeps all rows from one impression in the same split.
- filter_seen strategy: `{",".join(filter_seen_splits)}`
- candidate_pool_k: `{candidate_pool_k}`
- cutoffs: `{cutoffs}`
- retriever list: `{", ".join(retrievers)}`
- ranker list: `{", ".join(rankers)}`
- RRF source weights: `{json.dumps(source_weights, ensure_ascii=False, sort_keys=True)}`

## Candidate Recall Before Ranker

{_candidate_meta_table(candidate_meta)}

## Top Runs

{_df_to_markdown(top_rows)}

## Retriever Comparison

{_group_table(ok, ["retrievers"], "Recall@10")}

## Ranker Comparison

{_group_table(ok, ["ranker"], "Recall@10")}

## Rerank Fusion Comparison

{_group_table(ok, ["rerank_mode"], "Recall@10")}

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

{_group_table(ok, ["ranker", "rerank_mode"], "Recall@10")}

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

{_group_table(ok, ["retrievers"], "Coverage")}

## Business Rerank Before / After

{_business_rerank_table(ok)}

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

{_group_table(ok, ["retrievers", "business_reranker"], "TailRatio")}

## LLM Role and Safety

LLM is not allowed to generate recommendation item IDs. The measurable safety smoke test checks hallucinated item ID rejection.

| Metric | Value |
|---|---:|
| hallucinated_item_id_rate | {llm_safety['hallucinated_item_id_rate']:.6f} |
| template_fallback_rate | {llm_safety['template_fallback_rate']:.6f} |
| unsupported_claim_rate | {llm_safety['unsupported_claim_rate']:.6f} |

## Failure Cases

See `failure_cases.json` for users with no hit in the evaluated recommendation list.

## Known Limitations

- Offline metrics do not prove online CTR or retention lift.
- Compact neural retrievers/rankers are for interview-grade experiments, not distributed production training.
- Full KuaiRec requires local raw data; sample data is provided only for command smoke tests.
"""
    report_path.write_text(text, encoding="utf-8")


def _load_candidate_meta(features_dir: Path) -> dict[str, Any]:
    path = features_dir / "ranker_candidate_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_stats(candidate_meta: dict[str, Any], split: str) -> dict[str, object]:
    split_stats = candidate_meta.get("splits", {}) if isinstance(candidate_meta, dict) else {}
    if isinstance(split_stats, dict) and split in split_stats:
        return dict(split_stats[split])
    overall = candidate_meta.get("overall", {}) if isinstance(candidate_meta, dict) else {}
    return dict(overall) if isinstance(overall, dict) else {}


def _candidate_meta_table(candidate_meta: dict[str, Any]) -> str:
    split_stats = candidate_meta.get("splits", {}) if isinstance(candidate_meta, dict) else {}
    if not split_stats:
        return "No candidate-level ranker metadata found. Run `scripts/build_ranker_candidates.py` first."
    rows = []
    for split, stats in split_stats.items():
        rows.append(
            {
                "split": split,
                "rows": stats.get("rows", 0),
                "positives": stats.get("positives", 0),
                "retrieved_positives": stats.get("retrieved_positives", 0),
                "injected_positives": stats.get("injected_positives", 0),
                "candidate_positive_recall": stats.get("candidate_positive_recall", 0.0),
                "positive_injection_rate": stats.get("positive_injection_rate", 0.0),
            }
        )
    overall = candidate_meta.get("overall", {})
    if isinstance(overall, dict) and overall:
        rows.append(
            {
                "split": "overall",
                "rows": "",
                "positives": overall.get("positives", 0),
                "retrieved_positives": overall.get("retrieved_positives", 0),
                "injected_positives": overall.get("injected_positives", 0),
                "candidate_positive_recall": overall.get("candidate_positive_recall", 0.0),
                "positive_injection_rate": overall.get("positive_injection_rate", 0.0),
            }
        )
    return _df_to_markdown(pd.DataFrame(rows))


def _data_stats_table(meta: dict[str, object]) -> str:
    rows = [
        {"field": "num_users", "value": meta.get("num_users", "")},
        {"field": "num_items", "value": meta.get("num_items", "")},
        {"field": "num_interactions", "value": meta.get("num_interactions", "")},
        {"field": "positive_rate", "value": meta.get("positive_rate", "")},
        {"field": "split_stats", "value": meta.get("split_stats", "")},
        {"field": "is_subset", "value": meta.get("is_subset", "")},
        {"field": "is_sample_fallback", "value": meta.get("is_sample_fallback", "")},
        {"field": "is_simulated", "value": meta.get("is_simulated", "")},
        {"field": "data_origin", "value": meta.get("data_origin", "")},
        {"field": "simulation_manifest", "value": meta.get("simulation_manifest", "")},
    ]
    return _df_to_markdown(pd.DataFrame(rows))


def _label_definition(dataset: str) -> str:
    if dataset == "kuairec":
        return "Positive label: `watch_ratio >= positive_watch_ratio_threshold` after adapter normalization. Default threshold is `1.0`."
    if dataset == "movielens":
        return "Positive label: `rating >= positive_rating_threshold`. Default threshold is `4.0`."
    if dataset == "mind":
        return "Positive label: clicked impression rows from MIND behavior logs."
    return "Positive label follows the dataset adapter's normalized `label` column."


def _split_strategy(dataset: str, filter_seen_splits: list[str]) -> str:
    extra = " MIND keeps all rows from one impression in the same split." if dataset == "mind" else ""
    return (
        "Interactions are split per user by timestamp into train/val/test."
        f"{extra} Evaluation filters past positive items from `{','.join(filter_seen_splits)}`."
    )


def _cold_start_table(processed_dir: Path) -> str:
    interactions_path = processed_dir / "interactions.parquet"
    if not interactions_path.exists():
        return "No processed interactions found."
    interactions = pd.read_parquet(interactions_path)
    train_counts = {
        str(user_id): int(count)
        for user_id, count in (
            interactions[(interactions["split"] == "train") & (interactions["label"] == 1)]
            .groupby("user_id")
            .size()
            .items()
        )
    }
    test_users = sorted(
        interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]["user_id"]
        .astype(str)
        .unique()
    )
    buckets = {
        "new_user_0_train_pos": 0,
        "short_history_1_2": 0,
        "medium_history_3_9": 0,
        "warm_history_10_plus": 0,
    }
    for user_id in test_users:
        count = int(train_counts.get(user_id, 0))
        if count == 0:
            buckets["new_user_0_train_pos"] += 1
        elif count <= 2:
            buckets["short_history_1_2"] += 1
        elif count <= 9:
            buckets["medium_history_3_9"] += 1
        else:
            buckets["warm_history_10_plus"] += 1
    rows = [{"bucket": bucket, "test_positive_users": count} for bucket, count in buckets.items()]
    return _df_to_markdown(pd.DataFrame(rows))


def _tail_bucket_table(processed_dir: Path) -> str:
    interactions_path = processed_dir / "interactions.parquet"
    items_path = processed_dir / "items.parquet"
    if not interactions_path.exists() or not items_path.exists():
        return "No processed items/interactions found."
    interactions = pd.read_parquet(interactions_path)
    items = pd.read_parquet(items_path)
    popularity = {
        str(item_id): int(count)
        for item_id, count in (
            interactions[(interactions["split"] == "train") & (interactions["label"] == 1)]
            .groupby("item_id")
            .size()
            .items()
        )
    }
    positive_counts = [int(count) for count in popularity.values() if int(count) > 0]
    if positive_counts:
        q20 = pd.Series(positive_counts).quantile(0.2)
        q80 = pd.Series(positive_counts).quantile(0.8)
    else:
        q20 = q80 = 0
    test_positive_items = set(
        interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]["item_id"].astype(str)
    )
    buckets = {
        "cold_item_0_train_pos": {"catalog_items": 0, "test_positive_items": 0},
        "tail_item_low_pop": {"catalog_items": 0, "test_positive_items": 0},
        "mid_item": {"catalog_items": 0, "test_positive_items": 0},
        "head_item_high_pop": {"catalog_items": 0, "test_positive_items": 0},
    }
    for item_id in items["item_id"].astype(str):
        count = int(popularity.get(item_id, 0))
        if count == 0:
            bucket = "cold_item_0_train_pos"
        elif count <= q20:
            bucket = "tail_item_low_pop"
        elif count >= q80:
            bucket = "head_item_high_pop"
        else:
            bucket = "mid_item"
        buckets[bucket]["catalog_items"] += 1
        buckets[bucket]["test_positive_items"] += int(item_id in test_positive_items)
    rows = [{"bucket": bucket, **values} for bucket, values in buckets.items()]
    return _df_to_markdown(pd.DataFrame(rows))


def _failure_case_table(failure_cases: dict[str, object], limit: int = 12) -> str:
    rows = []
    for combo, cases in failure_cases.items():
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            rows.append(
                {
                    "combo": combo,
                    "user_id": case.get("user_id", ""),
                    "relevant_items": case.get("relevant_items", ""),
                    "recommended_items": case.get("recommended_items", ""),
                    "error": case.get("error", ""),
                }
            )
            if len(rows) >= limit:
                return _df_to_markdown(pd.DataFrame(rows))
    return "No failure cases found in the evaluated rows."


def _llm_table(llm_safety: dict[str, float]) -> str:
    rows = [{"metric": key, "value": value} for key, value in sorted(llm_safety.items())]
    return _df_to_markdown(pd.DataFrame(rows))


def _write_formal_report(
    metrics: pd.DataFrame,
    processed_dir: Path,
    report_path: Path,
    dataset: str,
    filter_seen_splits: list[str],
    candidate_pool_k: int,
    cutoffs: list[int],
    retrievers: list[str],
    rankers: list[str],
    candidate_meta: dict[str, Any],
    failure_cases: dict[str, object],
    llm_safety: dict[str, float],
    source_weights: dict[str, float],
) -> None:
    meta_path = processed_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ok = metrics[metrics["status"].astype(str).str.startswith("ok")] if "status" in metrics else metrics
    text = f"""# LLM4Rec Formal Experiment Report

Dataset: `{dataset}`

## Dataset Statistics

{_data_stats_table(meta)}

## Label Definition

{_label_definition(dataset)}

## Split Strategy

{_split_strategy(dataset, filter_seen_splits)}

## Candidate Recall Before Ranker

The ranker is trained on candidate-level samples generated by real retrievers and RRF fusion. Missing split positives are injected only as audited supervision rows; they are not exposed as a `label_positive` retrieval-source embedding.

{_candidate_meta_table(candidate_meta)}

## Positive Injection Rate

`positive_injection_rate = injected_positives / all_positive_labels` for each split. Lower is better because it means candidate recall already covers more positives before ranker training.

{_candidate_meta_table(candidate_meta)}

## Retriever Comparison

{_group_table(ok, ["retrievers"], "Recall@10")}

## Ranker Comparison

{_group_table(ok, ["ranker"], "Recall@10")}

## Fusion Comparison

{_group_table(ok, ["fusion", "rerank_mode"], "Recall@10")}

## Business Rerank Before / After

{_business_rerank_table(ok)}

## Cold-Start Buckets

{_cold_start_table(processed_dir)}

## Tail Item Buckets

{_tail_bucket_table(processed_dir)}

## Failure Cases

{_failure_case_table(failure_cases)}

## LLM Safety / Query Rewrite Experiment

LLM output is only allowed to rewrite query text or generate explanations. It is not allowed to generate recommendation item IDs.

{_llm_table(llm_safety)}

## Known Limitations

- Offline Recall/NDCG/MRR/MAP do not prove online CTR, watch-time, or retention lift.
- The local neural retrievers/rankers are compact interview implementations, not distributed production training jobs.
- Full KuaiRec results require real raw data under `data/raw/kuairec`; sample fallback is only a smoke-test path and is marked in `meta.json`.
- Positive injection makes ranker supervision possible when retrieval misses labels, but a high injection rate means the recall stage still needs improvement.
- Cold-start and tail analysis here are offline buckets, not production traffic cohorts.
- The LLM guardrail is a deterministic safety smoke test, not a complete red-team evaluation.

## Reproducibility

- candidate_pool_k: `{candidate_pool_k}`
- cutoffs: `{cutoffs}`
- retrievers: `{", ".join(retrievers)}`
- rankers: `{", ".join(rankers)}`
- RRF source weights: `{json.dumps(source_weights, ensure_ascii=False, sort_keys=True)}`
"""
    report_path.write_text(text, encoding="utf-8")


def _business_rerank_table(metrics: pd.DataFrame) -> str:
    columns = [
        "business_reranker",
        "NDCG@10_before_business_rerank",
        "NDCG@10_after_business_rerank",
        "Coverage_before_business_rerank",
        "Coverage_after_business_rerank",
        "Diversity_before_business_rerank",
        "Diversity_after_business_rerank",
        "TailRatio_before_business_rerank",
        "TailRatio_after_business_rerank",
    ]
    existing = [column for column in columns if column in metrics.columns]
    if not existing:
        return "No business rerank rows."
    table = metrics[existing].dropna(how="all").head(20)
    return _df_to_markdown(table) if not table.empty else "No business rerank rows."


def _ranker_metrics(ranker_root: Path, ranker: str) -> dict[str, float | None]:
    if ranker == "none":
        return {"AUC": None, "LogLoss": None}
    path = ranker_root / ranker / "metrics.json"
    if not path.exists():
        return {"AUC": None, "LogLoss": None}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {"AUC": raw.get("val_auc"), "LogLoss": raw.get("val_logloss")}


def _ensure_rankers(
    dataset: str,
    rankers: list[str],
    features_dir: Path,
    ranker_root: Path,
    ranking_cfg: dict[str, Any],
    features_cfg: dict[str, Any],
) -> None:
    if not features_dir.exists():
        return
    if not (features_dir / "train_rank_samples.parquet").exists():
        return
    epochs = int(ranking_cfg.get("epochs", 3))
    batch_size = int(ranking_cfg.get("batch_size", 256))
    embedding_dim = int(ranking_cfg.get("embedding_dim", 16))
    max_history_len = int(features_cfg.get("max_history_len", 20))
    for ranker in rankers:
        if ranker == "none":
            continue
        model_dir = ranker_root / ranker
        if ranker == "gbdt":
            if (model_dir / "model.pkl").exists() and (model_dir / "config.json").exists():
                continue
            gbdt_cfg = ranking_cfg.get("gbdt", {}) if isinstance(ranking_cfg.get("gbdt", {}), dict) else {}
            cmd = [
                sys.executable,
                "scripts/train_gbdt_ranker.py",
                "--dataset",
                dataset,
                "--features_dir",
                str(features_dir),
                "--output_dir",
                str(model_dir),
                "--train_split",
                str(gbdt_cfg.get("train_split", "train")),
                "--val_split",
                str(gbdt_cfg.get("val_split", "val")),
                "--max_iter",
                str(gbdt_cfg.get("max_iter", 80)),
                "--learning_rate",
                str(gbdt_cfg.get("learning_rate", 0.08)),
                "--max_leaf_nodes",
                str(gbdt_cfg.get("max_leaf_nodes", 31)),
                "--l2_regularization",
                str(gbdt_cfg.get("l2_regularization", 0.05)),
            ]
            if gbdt_cfg.get("max_train_rows") is not None:
                cmd.extend(["--max_train_rows", str(gbdt_cfg["max_train_rows"])])
            if gbdt_cfg.get("max_val_rows") is not None:
                cmd.extend(["--max_val_rows", str(gbdt_cfg["max_val_rows"])])
            print(f"Training missing ranker artifact: {ranker}")
            subprocess.run(cmd, check=True)
            continue
        if (model_dir / "model.pt").exists() and (model_dir / "config.json").exists():
            continue
        cmd = [
            sys.executable,
            "scripts/train_ranker.py",
            "--dataset",
            dataset,
            "--model",
            ranker,
            "--features_dir",
            str(features_dir),
            "--output_dir",
            str(model_dir),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--embedding_dim",
            str(embedding_dim),
            "--max_history_len",
            str(max_history_len),
            "--patience",
            "2",
        ]
        print(f"Training missing ranker artifact: {ranker}")
        subprocess.run(cmd, check=True)


def _llm_safety_eval() -> dict[str, float]:
    fallback = "template fallback"
    examples = [
        ("Recommend item_999 because it has 99999 likes.", ["item_1"], True),
        ("This selected item matches your interest.", ["item_1"], False),
        ("movie_42 is relevant.", ["movie_42"], False),
    ]
    query_rewrite_examples = [
        ("Rewrite query as sports highlights item_999", ["item_1"], True),
        ("Rewrite query as relaxing music videos", ["item_1"], False),
    ]
    hallucinated = 0
    fallback_count = 0
    unsupported_claim = 0
    for text, allowed, should_fallback in examples:
        guarded = reject_hallucinated_item_ids(text, allowed, fallback)
        did_fallback = guarded == fallback
        fallback_count += int(did_fallback)
        hallucinated += int(should_fallback)
        unsupported_claim += int("likes" in text.lower() and did_fallback)
    blocked_rewrites = 0
    unsafe_rewrites = 0
    for text, allowed, should_block in query_rewrite_examples:
        guarded = reject_hallucinated_item_ids(text, allowed, fallback)
        did_block = guarded == fallback
        blocked_rewrites += int(did_block)
        unsafe_rewrites += int(should_block)
    total = len(examples)
    return {
        "hallucinated_item_id_rate": hallucinated / total,
        "template_fallback_rate": fallback_count / total,
        "unsupported_claim_rate": unsupported_claim / total,
        "query_rewrite_id_guard_rate": blocked_rewrites / max(1, unsafe_rewrites),
        "query_rewrite_safe_pass_rate": (
            (len(query_rewrite_examples) - blocked_rewrites) / max(1, len(query_rewrite_examples) - unsafe_rewrites)
        ),
    }


def _retriever_sets(requested: list[str], configured: list[str]) -> list[tuple[str, list[str]]]:
    base = [name for name in configured if name != "fusion"]
    sets: list[tuple[str, list[str]]] = []
    for name in requested:
        if name == "fusion":
            sets.append(("fusion", base))
        else:
            sets.append((name, [name]))
    return sets


def _group_table(metrics: pd.DataFrame, group_cols: list[str], metric: str) -> str:
    if metrics.empty or metric not in metrics.columns:
        return "No successful runs."
    table = (
        metrics.groupby(group_cols, dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )
    return _df_to_markdown(table)


def _sort_metric(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    if metric not in metrics.columns:
        return metrics.head(0)
    return metrics.sort_values(metric, ascending=False, na_position="last")


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_cell(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6f}"
    return str(value)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or use a JSON config file.") from exc


def _split(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _parse_source_weights(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    weights: dict[str, float] = {}
    for part in _split(text):
        if "=" not in part:
            raise ValueError(f"Invalid source weight '{part}'. Expected name=value.")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


if __name__ == "__main__":
    main()
