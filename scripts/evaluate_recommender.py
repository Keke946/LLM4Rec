from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm4rec.data import load_user_behavior
from llm4rec.evaluation.evaluator import OfflineEvaluator, save_evaluation_outputs
from llm4rec.recommender import LLM4RecRecommender


def evaluate(k: int) -> dict[str, float | int]:
    behavior = load_user_behavior()
    recommender = LLM4RecRecommender()

    hits = 0
    total = 0
    tag_precisions: list[float] = []
    coverage: set[str] = set()

    for info in behavior.values():
        preferred = set(info["preferred_tags"])
        watched = set(info["watched_videos"])
        recs = recommender.recommend(list(preferred), k=k)
        rec_ids = {item["video_id"] for item in recs}
        hits += int(bool(watched & rec_ids))
        total += 1
        coverage.update(rec_ids)
        if recs:
            matched_tags = [
                bool(preferred & set(item["tags"]))
                for item in recs
            ]
            tag_precisions.append(sum(matched_tags) / len(matched_tags))

    return {
        "users": total,
        f"hit_rate@{k}": hits / total if total else 0.0,
        f"tag_precision@{k}": sum(tag_precisions) / len(tag_precisions) if tag_precisions else 0.0,
        "catalog_coverage": len(coverage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local recommendation quality.")
    parser.add_argument("--dataset", default=None, help="Dataset name for processed-data evaluation.")
    parser.add_argument("--processed_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--retrievers", default="popular,itemcf", help="Comma-separated retrievers.")
    parser.add_argument("--ranker", default=None, help="Reserved ranker argument for report compatibility.")
    parser.add_argument("--top_k", type=int, default=None, help="Deprecated alias for --final_top_k.")
    parser.add_argument("--candidate_pool_k", type=int, default=500, help="Merged candidate pool size.")
    parser.add_argument("--cutoffs", default="5,10,20,50,100", help="Comma-separated metric cutoffs.")
    parser.add_argument("--final_top_k", type=int, default=10, help="Final displayed recommendation count.")
    parser.add_argument("--filter_seen_splits", default="train,val", help="Comma-separated past positive splits to filter.")
    parser.add_argument(
        "--rerank_mode",
        default="retrieval_ranker_fusion",
        choices=["retrieval_only", "ranker_only", "retrieval_ranker_fusion"],
        help="How to combine retrieval and ranker scores.",
    )
    parser.add_argument("--retrieval_weight", type=float, default=0.7, help="Retrieval score weight.")
    parser.add_argument("--ranker_weight", type=float, default=0.3, help="Ranker probability weight.")
    parser.add_argument(
        "--source_weights",
        default=None,
        help="Optional comma-separated RRF source weights, e.g. sasrec=1.2,metadata=0.4.",
    )
    parser.add_argument(
        "--reranker",
        default="none",
        choices=["none", "diversity", "diversity_tail"],
        help="Optional business reranker after model ranking.",
    )
    parser.add_argument("--output_dir", default=None, help="Report output directory.")
    parser.add_argument("--k", type=int, default=5, help="Recommendation cutoff.")
    args = parser.parse_args()
    if args.dataset:
        processed_dir = args.processed_dir or f"data/processed/{args.dataset}"
        output_dir = args.output_dir or f"reports/{args.dataset}"
        retrievers = [name.strip() for name in args.retrievers.split(",") if name.strip()]
        evaluator = OfflineEvaluator(
            processed_dir=processed_dir,
            retriever_dir=f"models/{args.dataset}/retrievers",
            features_dir=f"data/features/{args.dataset}",
            ranker_root=f"models/{args.dataset}",
        )
        cutoffs = [int(value.strip()) for value in args.cutoffs.split(",") if value.strip()]
        filter_seen_splits = [
            value.strip()
            for value in args.filter_seen_splits.split(",")
            if value.strip()
        ]
        result = evaluator.evaluate_retrievers(
            retrievers,
            top_k=args.top_k,
            ranker_name=args.ranker,
            candidate_pool_k=args.candidate_pool_k,
            cutoffs=cutoffs,
            final_top_k=args.final_top_k,
            filter_seen_splits=filter_seen_splits,
            rerank_mode=args.rerank_mode,
            retrieval_weight=args.retrieval_weight,
            ranker_weight=args.ranker_weight,
            reranker=args.reranker,
            source_weights=_parse_source_weights(args.source_weights),
        )
        meta_path = Path(processed_dir) / "meta.json"
        if meta_path.exists():
            result["data_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        config_path = Path(f"models/{args.dataset}/{args.ranker}/config.json") if args.ranker else None
        if config_path and config_path.exists():
            result["model_config"] = json.loads(config_path.read_text(encoding="utf-8"))
        save_evaluation_outputs(result, output_dir, args.dataset)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved report artifacts to {output_dir}")
        return
    print(json.dumps(evaluate(args.k), ensure_ascii=False, indent=2))


def _parse_source_weights(text: str | None) -> dict[str, float]:
    if not text:
        return {}
    weights: dict[str, float] = {}
    for part in [value.strip() for value in text.split(",") if value.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid source weight '{part}'. Expected name=value.")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


if __name__ == "__main__":
    main()
