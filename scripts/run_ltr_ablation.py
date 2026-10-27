from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from llm4rec.evaluation.metrics import ranking_metrics, tail_ratio
from llm4rec.ranking.lambdarank import load_lambdarank_model, make_lambdarank_features
from llm4rec.scoring import compute_final_scores


MAIN_EXPERIMENTS = [
    {"experiment": "retrieval_only", "ranker": None, "rerank_mode": "retrieval_only", "ranker_root": "base"},
    {"experiment": "gbdt_full", "ranker": "gbdt", "rerank_mode": "ranker_only", "ranker_root": "base"},
    {"experiment": "gbdt_top50", "ranker": "gbdt", "rerank_mode": "ranker_top_50", "ranker_root": "base"},
    {"experiment": "gbdt_top100", "ranker": "gbdt", "rerank_mode": "ranker_top_100", "ranker_root": "base"},
    {"experiment": "lambdarank_full", "ranker": "lambdarank", "rerank_mode": "ranker_only", "ranker_root": "base"},
    {"experiment": "lambdarank_top50", "ranker": "lambdarank", "rerank_mode": "ranker_top_50", "ranker_root": "base"},
    {"experiment": "lambdarank_top100", "ranker": "lambdarank", "rerank_mode": "ranker_top_100", "ranker_root": "base"},
]

NEGATIVE_STRATEGIES = [
    "random_negative",
    "exposed_low_watch_negative",
    "retrieved_hard_negative",
    "mixed_negative",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KuaiRec LambdaRank/LTR ablation experiments.")
    parser.add_argument("--dataset", default="kuairec")
    parser.add_argument("--config", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--retriever_dir", default=None)
    parser.add_argument("--ranker_root", default=None)
    parser.add_argument("--negative_ranker_root", default=None)
    parser.add_argument("--negative_features_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--retrievers", default=None)
    parser.add_argument("--candidate_pool_k", type=int, default=None)
    parser.add_argument("--cutoffs", default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--filter_seen_splits", default=None)
    parser.add_argument("--retrieval_weight", type=float, default=None)
    parser.add_argument("--ranker_weight", type=float, default=None)
    parser.add_argument("--source_weights", default=None)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--negatives_per_positive", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    paths = _dict(config.get("paths"))
    evaluation_cfg = _dict(config.get("evaluation"))
    train_cfg = _dict(config.get("training"))
    negative_cfg = _dict(config.get("negative_sampling"))
    retrieval_cfg = _dict(config.get("retrieval"))

    dataset = args.dataset or str(config.get("dataset") or "kuairec")
    processed_dir = Path(args.processed_dir or paths.get("processed_dir") or f"data/processed/{dataset}")
    features_dir = Path(args.features_dir or paths.get("features_dir") or f"data/features/{dataset}_trend1500_ltr")
    retriever_dir = Path(args.retriever_dir or paths.get("retriever_dir") or f"models/{dataset}/retrievers")
    ranker_root = Path(args.ranker_root or paths.get("ranker_root") or f"models/{dataset}_trend1500_ltr")
    negative_ranker_root = Path(
        args.negative_ranker_root or paths.get("negative_ranker_root") or f"models/{dataset}_ltr_ablation"
    )
    negative_features_root = Path(
        args.negative_features_root or paths.get("negative_features_root") or f"data/features/{dataset}_ltr_ablation"
    )
    output_dir = Path(args.output_dir or paths.get("output_dir") or f"reports/{dataset}_ltr_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    retrievers = _split(args.retrievers) or _as_list(retrieval_cfg.get("retrievers")) or ["trend"]
    cutoffs = _int_list(args.cutoffs) or [int(value) for value in _as_list(evaluation_cfg.get("cutoffs"))] or [5, 10, 20, 50, 100]
    final_top_k = int(args.final_top_k or evaluation_cfg.get("final_top_k") or max(cutoffs))
    candidate_pool_k = int(args.candidate_pool_k or evaluation_cfg.get("candidate_pool_k") or 1500)
    filter_seen_splits = _split(args.filter_seen_splits) or _as_list(evaluation_cfg.get("filter_seen_splits")) or ["train", "val"]
    retrieval_weight = float(args.retrieval_weight or evaluation_cfg.get("retrieval_weight") or 0.7)
    ranker_weight = float(args.ranker_weight or evaluation_cfg.get("ranker_weight") or 0.3)
    source_weights = _parse_weights(args.source_weights) or {
        str(key): float(value) for key, value in _dict(retrieval_cfg.get("source_weights")).items()
    }
    n_estimators = int(args.n_estimators or train_cfg.get("n_estimators") or 120)
    max_train_rows = args.max_train_rows if args.max_train_rows is not None else train_cfg.get("max_train_rows")
    max_val_rows = args.max_val_rows if args.max_val_rows is not None else train_cfg.get("max_val_rows")
    negatives_per_positive = int(
        args.negatives_per_positive
        if args.negatives_per_positive is not None
        else negative_cfg.get("negatives_per_positive", 20)
    )
    negative_strategies = _as_list(negative_cfg.get("strategies")) or NEGATIVE_STRATEGIES

    if not args.skip_training:
        _ensure_lambdarank(
            dataset,
            features_dir,
            ranker_root / "lambdarank",
            n_estimators=n_estimators,
            max_train_rows=max_train_rows,
            max_val_rows=max_val_rows,
        )
        for strategy in negative_strategies:
            _ensure_negative_lambdarank(
                dataset,
                processed_dir,
                features_dir,
                negative_features_root / strategy,
                negative_ranker_root / f"lambdarank_{strategy}",
                output_dir / strategy,
                strategy=strategy,
                n_estimators=n_estimators,
                max_train_rows=max_train_rows,
                max_val_rows=max_val_rows,
                negatives_per_positive=negatives_per_positive,
            )

    frame_cache: dict[str, pd.DataFrame] = {}
    score_cache: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    selected_failure_cases: list[dict[str, Any]] = []
    for experiment in MAIN_EXPERIMENTS:
        root = ranker_root if experiment["ranker_root"] == "base" else negative_ranker_root
        row, result = _evaluate_one(
            dataset,
            processed_dir,
            retriever_dir,
            features_dir,
            root,
            frame_cache,
            score_cache,
            experiment["experiment"],
            retrievers,
            experiment["ranker"],
            experiment["rerank_mode"],
            candidate_pool_k,
            cutoffs,
            final_top_k,
            filter_seen_splits,
            retrieval_weight,
            ranker_weight,
            source_weights,
        )
        rows.append(row)
        if experiment["experiment"] == "lambdarank_top50" and result is not None:
            selected_failure_cases = list(result.get("failure_cases", []))

    negative_rows: list[dict[str, Any]] = []
    for strategy in negative_strategies:
        experiment_name = f"lambdarank_{strategy}"
        row, _ = _evaluate_one(
            dataset,
            processed_dir,
            retriever_dir,
            features_dir,
            negative_ranker_root,
            frame_cache,
            score_cache,
            experiment_name,
            retrievers,
            experiment_name,
            "ranker_top_50",
            candidate_pool_k,
            cutoffs,
            final_top_k,
            filter_seen_splits,
            retrieval_weight,
            ranker_weight,
            source_weights,
        )
        row["negative_strategy"] = strategy
        negative_rows.append(row)
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_summary.md").write_text(_metrics_markdown(metrics), encoding="utf-8")
    _write_failure_outputs(output_dir, selected_failure_cases)
    _write_negative_outputs(output_dir, negative_rows, negative_strategies)
    _write_final_report(
        output_dir,
        Path("reports") / "FINAL_EXPERIMENT_SUMMARY_V2.md",
        dataset,
        processed_dir,
        features_dir,
        metrics,
        negative_rows,
        cutoffs,
        candidate_pool_k,
        filter_seen_splits,
    )
    print(f"Saved LTR ablation outputs to {output_dir}")


def _ensure_lambdarank(
    dataset: str,
    features_dir: Path,
    output_dir: Path,
    n_estimators: int,
    max_train_rows: Any,
    max_val_rows: Any,
) -> None:
    if (output_dir / "model.pkl").exists() and (output_dir / "feature_schema.json").exists():
        return
    cmd = [
        sys.executable,
        "scripts/train_lambdarank_ranker.py",
        "--dataset",
        dataset,
        "--features_dir",
        str(features_dir),
        "--output_dir",
        str(output_dir),
        "--n_estimators",
        str(n_estimators),
    ]
    if max_train_rows is not None:
        cmd.extend(["--max_train_rows", str(max_train_rows)])
    if max_val_rows is not None:
        cmd.extend(["--max_val_rows", str(max_val_rows)])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _ensure_negative_lambdarank(
    dataset: str,
    processed_dir: Path,
    input_features_dir: Path,
    output_features_dir: Path,
    output_model_dir: Path,
    report_dir: Path,
    strategy: str,
    n_estimators: int,
    max_train_rows: Any,
    max_val_rows: Any,
    negatives_per_positive: int,
) -> None:
    if not (output_features_dir / "train_rank_samples.parquet").exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/build_exposure_ranker_train.py",
                "--dataset",
                dataset,
                "--processed_dir",
                str(processed_dir),
                "--input_dir",
                str(input_features_dir),
                "--output_dir",
                str(output_features_dir),
                "--report_dir",
                str(report_dir),
                "--experiment_name",
                report_dir.name,
                "--strategy",
                strategy,
                "--negatives_per_positive",
                str(negatives_per_positive),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    _ensure_lambdarank(
        dataset,
        output_features_dir,
        output_model_dir,
        n_estimators=n_estimators,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
    )


def _evaluate_one(
    dataset: str,
    processed_dir: Path,
    retriever_dir: Path,
    features_dir: Path,
    ranker_root: Path,
    frame_cache: dict[str, pd.DataFrame],
    score_cache: dict[str, pd.Series],
    experiment: str,
    retrievers: list[str],
    ranker: str | None,
    rerank_mode: str,
    candidate_pool_k: int,
    cutoffs: list[int],
    final_top_k: int,
    filter_seen_splits: list[str],
    retrieval_weight: float,
    ranker_weight: float,
    source_weights: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        result = _evaluate_candidate_pool(
            processed_dir=processed_dir,
            features_dir=features_dir,
            ranker_root=ranker_root,
            frame_cache=frame_cache,
            score_cache=score_cache,
            ranker=ranker,
            rerank_mode=rerank_mode,
            candidate_pool_k=candidate_pool_k,
            cutoffs=cutoffs,
            retrieval_weight=retrieval_weight,
            ranker_weight=ranker_weight,
        )
        summary = dict(result.get("summary", {}))
        status = "ok" if ranker is None or result.get("ranker_available") else "ranker_missing_or_skipped"
        return {
            "experiment": experiment,
            "dataset": dataset,
            "retrievers": ",".join(retrievers),
            "ranker": ranker or "none",
            "rerank_mode": rerank_mode,
            "candidate_pool_k": candidate_pool_k,
            "status": status,
            **summary,
            **_candidate_stats(features_dir),
        }, result
    except Exception as exc:
        return {
            "experiment": experiment,
            "dataset": dataset,
            "retrievers": ",".join(retrievers),
            "ranker": ranker or "none",
            "rerank_mode": rerank_mode,
            "candidate_pool_k": candidate_pool_k,
            "status": f"failed: {type(exc).__name__}: {exc}",
        }, None


def _evaluate_candidate_pool(
    processed_dir: Path,
    features_dir: Path,
    ranker_root: Path,
    frame_cache: dict[str, pd.DataFrame],
    score_cache: dict[str, pd.Series],
    ranker: str | None,
    rerank_mode: str,
    candidate_pool_k: int,
    cutoffs: list[int],
    retrieval_weight: float,
    ranker_weight: float,
) -> dict[str, Any]:
    full = _load_test_candidates(features_dir, frame_cache)
    if full.empty:
        raise RuntimeError(f"No test candidate rows found under {features_dir}")
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    relevant_by_user = {
        str(user): set(group["item_id"].astype(str))
        for user, group in interactions[(interactions["split"] == "test") &
                                       (interactions["label"] == 1)].groupby("user_id")
    }
    live = _live_candidate_rows(full).copy()
    live = live.sort_values(["user_id", "rank", "item_id"]).groupby("user_id", group_keys=False).head(candidate_pool_k)
    ranker_available = False
    if ranker and rerank_mode != "retrieval_only":
        scores = _ranker_scores(live, ranker_root, ranker, score_cache)
        if scores is None:
            raise FileNotFoundError(f"Requested ranker {ranker!r} is missing.")
        if scores is not None:
            live = live.copy()
            live["_ranker_score"] = scores.reindex(live.index).fillna(0.0)
            ranker_available = True
    recommended_by_user: dict[str, list[str]] = {}
    for user_id, group in live.groupby("user_id", sort=False):
        ordered = _ordered_items_for_group(group, rerank_mode, retrieval_weight, ranker_weight)
        recommended_by_user[str(user_id)] = ordered[: max(cutoffs)]
    per_user = [
        ranking_metrics(recommended_by_user.get(user_id, []), relevant, cutoffs)
        for user_id, relevant in relevant_by_user.items()
        if relevant
    ]
    summary = _mean_metrics(per_user)
    train_popularity, tail_items = _train_popularity(processed_dir)
    summary["TailRatio"] = tail_ratio(recommended_by_user, tail_items)
    summary["avg_candidates"] = float(live.groupby("user_id").size().mean()) if not live.empty else 0.0
    failure_cases = _candidate_failure_cases(
        live,
        recommended_by_user,
        relevant_by_user,
        train_popularity,
        tail_items,
    )
    return {
        "summary": summary,
        "ranker_available": ranker_available if ranker else False,
        "failure_cases": failure_cases,
    }


def _load_test_candidates(features_dir: Path, frame_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = features_dir / "test_rank_samples.parquet"
    key = str(path.resolve())
    if key not in frame_cache:
        frame_cache[key] = pd.read_parquet(path).reset_index(drop=True)
    return frame_cache[key]


def _live_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "is_injected_positive" not in frame.columns:
        return frame
    injected = pd.to_numeric(frame["is_injected_positive"], errors="coerce").fillna(0)
    return frame[injected <= 0]


def _ranker_scores(
    live: pd.DataFrame,
    ranker_root: Path,
    ranker: str,
    score_cache: dict[str, pd.Series],
) -> pd.Series | None:
    key = f"{ranker_root.resolve()}::{ranker}"
    if key in score_cache:
        return score_cache[key]
    model_dir = ranker_root / ranker
    if not model_dir.exists():
        return None
    if (model_dir / "feature_schema.json").exists() and (model_dir / "model.pkl").exists():
        model, schema = load_lambdarank_model(model_dir)
        scores = pd.Series(model.predict(make_lambdarank_features(live, schema)), index=live.index)
        score_cache[key] = scores
        return scores
    if ranker == "gbdt" and (model_dir / "model.pkl").exists() and (model_dir / "config.json").exists():
        with (model_dir / "model.pkl").open("rb") as file:
            model = pickle.load(file)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        feature_columns = [str(column) for column in config.get("feature_columns", [])]
        schema = {
            "source_names": [str(name) for name in config.get("source_names", [])],
            "feature_columns": feature_columns,
        }
        features = make_lambdarank_features(live, schema)
        scores = pd.Series(model.predict_proba(features[feature_columns].fillna(0.0))[:, 1], index=live.index)
        score_cache[key] = scores
        return scores
    return None


def _ordered_items_for_group(
    group: pd.DataFrame,
    rerank_mode: str,
    retrieval_weight: float,
    ranker_weight: float,
) -> list[str]:
    base = group.sort_values(["rank", "item_id"]).copy()
    if "_ranker_score" not in base.columns or rerank_mode == "retrieval_only":
        return base["item_id"].astype(str).tolist()
    head_k = _head_rerank_k(rerank_mode)
    if head_k is not None:
        head = base.head(head_k).sort_values(["_ranker_score", "rank"], ascending=[False, True])
        tail = base.iloc[head_k:]
        return pd.concat([head, tail])["item_id"].astype(str).tolist()
    if rerank_mode == "retrieval_ranker_fusion":
        retrieval_scores = pd.to_numeric(base.get("retrieval_score", 0.0), errors="coerce").fillna(0.0).tolist()
        ranker_scores = pd.to_numeric(base["_ranker_score"], errors="coerce").fillna(0.0).tolist()
        base["_final_score"] = compute_final_scores(
            retrieval_scores,
            ranker_scores,
            mode=rerank_mode,
            retrieval_weight=retrieval_weight,
            ranker_weight=ranker_weight,
        )
        return base.sort_values(["_final_score", "rank"], ascending=[False, True])["item_id"].astype(str).tolist()
    return base.sort_values(["_ranker_score", "rank"], ascending=[False, True])["item_id"].astype(str).tolist()


def _head_rerank_k(mode: str) -> int | None:
    prefix = "ranker_top_"
    if not str(mode).startswith(prefix):
        return None
    try:
        return max(1, int(str(mode)[len(prefix) :]))
    except ValueError:
        return None


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


def _train_popularity(processed_dir: Path) -> tuple[Counter[str], set[str]]:
    path = processed_dir / "interactions.parquet"
    if not path.exists():
        return Counter(), set()
    interactions = pd.read_parquet(path, columns=["item_id", "label", "split"])
    train = interactions[(interactions["split"] == "train") & (interactions["label"] == 1)]
    popularity = Counter(train["item_id"].astype(str))
    counts = sorted(popularity.values())
    threshold = counts[int(len(counts) * 0.2)] if counts else 0
    tail_items = {item for item, count in popularity.items() if count <= threshold}
    return popularity, tail_items


def _candidate_failure_cases(
    live: pd.DataFrame,
    recommended_by_user: dict[str, list[str]],
    relevant_by_user: dict[str, set[str]],
    popularity: Counter[str],
    tail_items: set[str],
    limit: int = 50,
) -> list[dict[str, Any]]:
    retrieved_by_user = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in live.groupby("user_id")
    }
    live_groups = {str(user_id): group for user_id, group in live.groupby("user_id")}
    candidate_counts = {
        str(user_id): int(len(group))
        for user_id, group in live.groupby("user_id")
    }
    cases: list[dict[str, Any]] = []
    for user_id, relevant in relevant_by_user.items():
        top10 = recommended_by_user.get(user_id, [])[:10]
        if set(top10) & relevant:
            continue
        retrieved = retrieved_by_user.get(user_id, set())
        positive_retrieved = retrieved & relevant
        user_group = live_groups.get(user_id, live.head(0))
        cases.append(
            {
                "user_id": user_id,
                "ground_truth_items": sorted(relevant)[:10],
                "retrieved_candidates_count": candidate_counts.get(user_id, 0),
                "positive_retrieved_count": len(positive_retrieved),
                "top10_items": top10,
                "miss_reason": _simple_miss_reason(
                    relevant,
                    top10,
                    positive_retrieved,
                    popularity,
                    tail_items,
                    user_group,
                ),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def _simple_miss_reason(
    relevant: set[str],
    top10: list[str],
    positive_retrieved: set[str],
    popularity: Counter[str],
    tail_items: set[str],
    user_group: pd.DataFrame,
) -> str:
    if not positive_retrieved:
        return "not_retrieved"
    if "history_len" in user_group.columns:
        history_len = pd.to_numeric(user_group["history_len"], errors="coerce").fillna(0).max()
        if history_len < 2:
            return "insufficient_user_history"
    relevant_pop = [int(popularity.get(item, 0)) for item in relevant]
    if relevant_pop and min(relevant_pop) == 0:
        return "cold_item"
    if relevant and all(item in tail_items for item in relevant):
        return "tail_item"
    top_pop = [int(popularity.get(item, 0)) for item in top10]
    if top_pop and relevant_pop:
        values = list(popularity.values())
        high = pd.Series(values).quantile(0.8) if values else 0
        low = pd.Series(values).quantile(0.2) if values else 0
        if sum(top_pop) / len(top_pop) >= high and sum(relevant_pop) / len(relevant_pop) <= low:
            return "popularity_bias"
    if "category_overlap" in user_group.columns:
        top_rows = user_group[user_group["item_id"].astype(str).isin(top10)]
        positive_rows = user_group[user_group["item_id"].astype(str).isin(positive_retrieved)]
        top_overlap = pd.to_numeric(top_rows["category_overlap"], errors="coerce").fillna(0).sum()
        positive_overlap = pd.to_numeric(positive_rows["category_overlap"], errors="coerce").fillna(0).sum()
        if positive_overlap > 0 and top_overlap == 0:
            return "category_mismatch"
    return "retrieved_but_low_rank"


def _candidate_stats(features_dir: Path) -> dict[str, Any]:
    path = features_dir / "ranker_candidate_meta.json"
    if not path.exists():
        return {"candidate_positive_recall": None, "positive_injection_rate": None}
    meta = json.loads(path.read_text(encoding="utf-8"))
    split_stats = _dict(_dict(meta.get("splits")).get("test"))
    if not split_stats:
        split_stats = _dict(meta.get("overall"))
    return {
        "candidate_positive_recall": split_stats.get("candidate_positive_recall"),
        "positive_injection_rate": split_stats.get("positive_injection_rate"),
    }


def _metrics_markdown(metrics: pd.DataFrame) -> str:
    ok = metrics[metrics["status"].astype(str).str.startswith("ok")] if "status" in metrics else metrics
    sort_metric = "NDCG@10" if "NDCG@10" in ok.columns else "Recall@10"
    if sort_metric in ok.columns:
        table = ok.sort_values(sort_metric, ascending=False, na_position="last")
    else:
        table = ok
    columns = [
        "experiment",
        "status",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
        "Recall@100",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "NDCG@50",
        "NDCG@100",
        "MRR@10",
        "MAP@10",
        "TailRatio",
        "avg_candidates",
        "candidate_positive_recall",
        "positive_injection_rate",
    ]
    return "# KuaiRec LTR Ablation Metrics\n\n" + _df_to_markdown(table[[column for column in columns if column in table.columns]]) + "\n"


def _write_failure_outputs(output_dir: Path, failure_cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in failure_cases:
        rows.append(
            {
                "user_id": case.get("user_id", ""),
                "ground_truth_items": json.dumps(case.get("ground_truth_items", case.get("relevant_items", [])), ensure_ascii=False),
                "retrieved_candidates_count": case.get("retrieved_candidates_count", 0),
                "positive_retrieved_count": case.get("positive_retrieved_count", 0),
                "top10_items": json.dumps(case.get("top10_items", case.get("recommended_items", [])), ensure_ascii=False),
                "miss_reason": case.get("miss_reason", ""),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "failure_cases.csv", index=False)
    expected_reasons = [
        "not_retrieved",
        "retrieved_but_low_rank",
        "cold_item",
        "tail_item",
        "popularity_bias",
        "category_mismatch",
        "insufficient_user_history",
    ]
    raw_counts = frame["miss_reason"].value_counts().to_dict() if not frame.empty else {}
    counts = pd.DataFrame(
        [{"miss_reason": reason, "count": int(raw_counts.get(reason, 0))} for reason in expected_reasons]
    )
    text = "# Failure Case Summary\n\n"
    text += "Failure cases are sampled from the LambdaRank top50 run when available.\n\n"
    text += _df_to_markdown(counts) + "\n"
    (output_dir / "failure_case_summary.md").write_text(text, encoding="utf-8")


def _write_negative_outputs(output_dir: Path, negative_rows: list[dict[str, Any]], strategies: list[str]) -> None:
    frame = pd.DataFrame(negative_rows)
    columns = ["negative_strategy", "status", "Recall@10", "NDCG@10", "MRR@10", "Recall@100"]
    text = "# Negative Sampling Ablation\n\n"
    text += _df_to_markdown(frame[[column for column in columns if column in frame.columns]]) + "\n"
    (Path("reports") / "negative_sampling_ablation.md").write_text(text, encoding="utf-8")
    mixed_stats = output_dir / "mixed_negative" / "negative_sampling_stats.json"
    if mixed_stats.exists():
        shutil.copy2(mixed_stats, output_dir / "negative_sampling_stats.json")
        mixed_md = output_dir / "mixed_negative" / "negative_sampling_stats.md"
        if mixed_md.exists():
            shutil.copy2(mixed_md, output_dir / "negative_sampling_stats.md")
    elif strategies:
        aggregate = {"strategies": strategies, "note": "No per-strategy negative sampling stats were available."}
        (output_dir / "negative_sampling_stats.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        (output_dir / "negative_sampling_stats.md").write_text("# Negative Sampling Stats\n\nNo stats available.\n", encoding="utf-8")


def _write_final_report(
    output_dir: Path,
    report_path: Path,
    dataset: str,
    processed_dir: Path,
    features_dir: Path,
    metrics: pd.DataFrame,
    negative_rows: list[dict[str, Any]],
    cutoffs: list[int],
    candidate_pool_k: int,
    filter_seen_splits: list[str],
) -> None:
    meta = _read_json(processed_dir / "meta.json")
    candidate_meta = _read_json(features_dir / "ranker_candidate_meta.json")
    main = metrics[metrics["experiment"].isin([row["experiment"] for row in MAIN_EXPERIMENTS])]
    negative = pd.DataFrame(negative_rows)
    result_columns = [
        "experiment",
        "status",
        "Recall@10",
        "NDCG@10",
        "MRR@10",
        "Recall@100",
        "candidate_positive_recall",
    ]
    negative_columns = ["negative_strategy", "status", "Recall@10", "NDCG@10", "MRR@10"]
    text = f"""# Final Experiment Summary V2

## 1. Project Goal

This project is a KuaiRec short-video multi-stage recommendation experiment. The goal is to move the repo from a demo-style prototype toward a real-data recommendation algorithm project with retriever recall, candidate fusion, temporal ranker training, reranking, evaluation, and diagnostics. LLM components are not used to directly choose recommendation item IDs.

## 2. Dataset and Split

- dataset: `{dataset}`
- users/items/interactions: `{meta.get('num_users', 'unknown')}` / `{meta.get('num_items', 'unknown')}` / `{meta.get('num_interactions', 'unknown')}`
- positive label definition: KuaiRec positive labels follow the processed adapter, typically `watch_ratio >= 1.0`.
- temporal split: per-user timestamp split into train/val/test.
- seen filter: `{','.join(filter_seen_splits)}`
- candidate pool: RRF over configured retrievers, candidate_pool_k=`{candidate_pool_k}`, cutoffs=`{cutoffs}`.

## 3. Pipeline

Raw data -> features -> multi-retriever recall -> candidate pool fusion -> ranker training -> rerank -> evaluation -> diagnostics.

## 4. Main Bottleneck Found

- candidate positive recall can be low before ranking.
- topK and cutoff definitions need to stay separate from final display topK.
- candidate pools shrink after seen filtering.
- injected positives are audit/supervision rows and can contaminate ranker training if treated as live retrieved candidates.
- pointwise full rerank can hurt deeper Recall@50/100, so head rerank is tracked separately.

## 5. Fixes Implemented

- candidate pool audit and candidate_positive_recall reporting
- train/val seen filter
- temporal ranker train candidates
- Trend and ExposureQuality retrievers in the existing pipeline
- GBDT head rerank baseline
- LightGBM LambdaRank ranker with fixed feature_schema.json
- exposure-aware negative sampling with random, exposed low-watch, retrieved hard, popularity hard, and mixed strategies

## 6. Main Results

{_df_to_markdown(main[[column for column in result_columns if column in main.columns]])}

## 7. Negative Sampling Ablation

{_df_to_markdown(negative[[column for column in negative_columns if column in negative.columns]])}

## 8. Rerank Scope Ablation

The matrix includes full ranker-only rerank and `ranker_top_50` / `ranker_top_100` head rerank. Head rerank can be stronger at this stage because it preserves retrieval recall depth while letting the ranker reorder the most plausible head candidates.

## 9. Failure Case Analysis

See `{output_dir / 'failure_cases.csv'}` and `{output_dir / 'failure_case_summary.md'}`. Failure reasons distinguish `not_retrieved`, `retrieved_but_low_rank`, `cold_item`, `tail_item`, `popularity_bias`, `category_mismatch`, and `insufficient_user_history`.

## 10. What Can Be Claimed

- built a real-data multi-stage recommendation pipeline
- implemented temporal split and candidate-level audit
- diagnosed recall bottleneck and ranking bottleneck
- improved or compared TopK ranking metrics through temporal LTR/head rerank where metrics support it
- analyzed exposure-aware negatives and failure cases

## 11. What Cannot Be Claimed

- not production-grade
- not SOTA
- no online A/B test
- no online CTR/watch-time lift
- LLM does not directly select recommendation item ids
- results are on the processed KuaiRec subset unless the metadata explicitly marks full data

## 12. Next Steps

- stronger sequence ranker
- better content semantic features
- KuaiRand/random exposure debiasing
- larger scale experiment
- serving demo only after metrics are stable

## Candidate Audit Snapshot

```json
{json.dumps(candidate_meta.get('overall', candidate_meta.get('splits', {})), ensure_ascii=False, indent=2)}
```
"""
    report_path.write_text(text, encoding="utf-8")


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text) or {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _split(text: str | None) -> list[str]:
    if not text:
        return []
    return [value.strip() for value in str(text).split(",") if value.strip()]


def _int_list(text: str | None) -> list[int]:
    return [int(value) for value in _split(text)]


def _parse_weights(text: str | None) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in _split(text):
        if "=" not in part:
            raise ValueError(f"Invalid source weight: {part}")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


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


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.6f}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


if __name__ == "__main__":
    main()
