# LLM4Rec Experiment Matrix

Dataset: `kuairec` (subset)

## Dataset Statistics

```json
{
  "dataset": "kuairec",
  "num_users": 1000,
  "num_items": 7483,
  "num_interactions": 300000,
  "positive_rate": 0.32984,
  "split_stats": {
    "test": 30006,
    "train": 239988,
    "val": 30006
  },
  "item_fields": [
    "item_id",
    "title",
    "description",
    "categories",
    "tags",
    "author_id",
    "duration",
    "full_text"
  ],
  "interaction_fields": [
    "user_id",
    "item_id",
    "timestamp",
    "label",
    "watch_ratio",
    "impression_id",
    "source",
    "split"
  ],
  "is_subset": true,
  "subset_info": {
    "sample_users": 1000,
    "max_interactions": 300000,
    "max_interactions_strategy": "random",
    "max_interactions_seed": 42
  },
  "is_sample_fallback": false,
  "data_origin": "data\\raw\\kuairec_official\\extracted\\KuaiRec 2.0\\data",
  "is_simulated": false,
  "simulation_manifest": null
}
```

## Experiment Setup

- Positive label definition: KuaiRec watch ratio threshold, MovieLens rating threshold, or MIND clicked impression labels depending on adapter.
- Split strategy: per-user temporal split; MIND additionally keeps all rows from one impression in the same split.
- filter_seen strategy: `train,val`
- candidate_pool_k: `1500`
- cutoffs: `[5, 10, 20, 50, 100]`
- retriever list: `trend, exposure`
- ranker list: `none, gbdt`
- RRF source weights: `{"exposure": 0.8, "trend": 1.2}`

## Candidate Recall Before Ranker

| split | rows | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | 1477304 | 9162 | 7858 | 1304 | 0.857673 | 0.142327 |
| test | 1482785 | 9287 | 7002 | 2285 | 0.753957 | 0.246043 |
| overall |  | 18449 | 14860 | 3589 | 0.805464 | 0.194536 |

## Top Runs

| dataset | retrievers | fusion | ranker | rerank_mode | business_reranker | candidate_pool_k | final_top_k | source_weights | candidate_positive_recall | positive_injection_rate | status | AUC | LogLoss | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank | HitRate@10 | HitRate@100 | HitRate@20 | HitRate@5 | HitRate@50 | MAP@10 | MRR@10 | NDCG@10 | NDCG@100 | NDCG@20 | NDCG@5 | NDCG@50 | Precision@10 | Precision@100 | Precision@20 | Precision@5 | Precision@50 | Recall@10 | Recall@100 | Recall@20 | Recall@5 | Recall@50 | CatalogCoverage | ItemCoverage | AveragePopularity | TailRatio | Diversity | Novelty | avg_candidates | latency_p50_ms | latency_p95_ms | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairec | fusion | RRF | gbdt | ranker_only | none | 1500 | 10 | {"exposure": 0.8, "trend": 1.2} | 0.753957 | 0.246043 | ok | 0.741960 | 0.543595 | 0.016110 | None | 0.044501 | None | 0.173404 | None | 0.000041 | None | 0.122594 | 0.682877 | 0.224924 | 0.063830 | 0.478217 | 0.005656 | 0.040873 | 0.016110 | 0.064116 | 0.022112 | 0.014796 | 0.041334 | 0.013678 | 0.013445 | 0.013576 | 0.013576 | 0.014245 | 0.015151 | 0.147232 | 0.030437 | 0.007556 | 0.078784 | 0.044501 | 333 | 30.241337 | 0.000041 | 0.173404 | 0.226538 | 1500.000000 | 13.127800 | 123.176320 | 0.044501 |
| kuairec | fusion | RRF | none | retrieval_only | none | 1500 | 10 | {"exposure": 0.8, "trend": 1.2} | 0.753957 | 0.246043 | ok |  |  | 0.011719 | None | 0.017105 | None | 0.168946 | None | 0.000000 | None | 0.107396 | 0.705167 | 0.238095 | 0.052685 | 0.487335 | 0.003349 | 0.025614 | 0.011719 | 0.062341 | 0.019810 | 0.009188 | 0.039100 | 0.011955 | 0.013556 | 0.014235 | 0.010740 | 0.014286 | 0.012416 | 0.150930 | 0.031403 | 0.005093 | 0.080406 | 0.017105 | 128 | 28.105562 | 0.000000 | 0.168946 | 0.234538 | 1500.000000 | 9.207500 | 99.749920 | 0.017105 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| fusion | 0.013783 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| gbdt | 0.015151 |
| none | 0.012416 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| ranker_only | 0.015151 |
| retrieval_only | 0.012416 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| gbdt | ranker_only | 0.015151 |
| none | retrieval_only | 0.012416 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| fusion | 0.030803 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.011719 | None | 0.017105 | None | 0.168946 | None | 0.000000 | None |
| none | 0.016110 | None | 0.044501 | None | 0.173404 | None | 0.000041 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| fusion | none | 0.000020 |

## LLM Role and Safety

LLM is not allowed to generate recommendation item IDs. The measurable safety smoke test checks hallucinated item ID rejection.

| Metric | Value |
|---|---:|
| hallucinated_item_id_rate | 0.333333 |
| template_fallback_rate | 0.333333 |
| unsupported_claim_rate | 0.333333 |

## Failure Cases

See `failure_cases.json` for users with no hit in the evaluated recommendation list.

## Known Limitations

- Offline metrics do not prove online CTR or retention lift.
- Compact neural retrievers/rankers are for interview-grade experiments, not distributed production training.
- Full KuaiRec requires local raw data; sample data is provided only for command smoke tests.
