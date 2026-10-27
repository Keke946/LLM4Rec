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
- retriever list: `popular, trend, itemcf, bpr, lightgcn, two_tower, sasrec, content_faiss, metadata`
- ranker list: `none, gbdt`
- RRF source weights: `{"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7}`

## Candidate Recall Before Ranker

| split | rows | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- |
| val | 1477371 | 9162 | 7791 | 1371 | 0.850360 | 0.149640 |
| test | 1482954 | 9287 | 6833 | 2454 | 0.735760 | 0.264240 |
| train | 1394026 | 14171 | 7645 | 6526 | 0.539482 | 0.460518 |
| overall |  | 32620 | 22269 | 10351 | 0.682679 | 0.317321 |

## Top Runs

| dataset | retrievers | fusion | ranker | rerank_mode | business_reranker | candidate_pool_k | final_top_k | source_weights | candidate_positive_recall | positive_injection_rate | status | AUC | LogLoss | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank | HitRate@10 | HitRate@100 | HitRate@20 | HitRate@5 | HitRate@50 | MAP@10 | MRR@10 | NDCG@10 | NDCG@100 | NDCG@20 | NDCG@5 | NDCG@50 | Precision@10 | Precision@100 | Precision@20 | Precision@5 | Precision@50 | Recall@10 | Recall@100 | Recall@20 | Recall@5 | Recall@50 | CatalogCoverage | ItemCoverage | AveragePopularity | TailRatio | Diversity | Novelty | avg_candidates | latency_p50_ms | latency_p95_ms | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairec | trend | single | none | retrieval_only | none | 1500 | 10 | {"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7} | 0.735760 | 0.264240 | ok |  |  | 0.010996 | None | 0.017907 | None | 0.169200 | None | 0.000000 | None | 0.102330 | 0.688956 | 0.192503 | 0.045593 | 0.483283 | 0.003112 | 0.023785 | 0.010996 | 0.060328 | 0.016457 | 0.008027 | 0.037443 | 0.011246 | 0.013040 | 0.011601 | 0.009321 | 0.013921 | 0.011924 | 0.149920 | 0.025259 | 0.004330 | 0.078487 | 0.017907 | 134 | 30.127042 | 0.000000 | 0.169200 | 0.220653 | 1500.000000 | 7.419000 | 13.480380 | 0.017907 |
| kuairec | trend | single | gbdt | retrieval_ranker_fusion | none | 1500 | 10 | {"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7} | 0.735760 | 0.264240 | ok | 0.663816 | 0.584799 | 0.011240 | None | 0.018308 | None | 0.169200 | None | 0.000000 | None | 0.102330 | 0.687943 | 0.192503 | 0.046606 | 0.482270 | 0.003253 | 0.024937 | 0.011240 | 0.060935 | 0.016660 | 0.008433 | 0.037769 | 0.011246 | 0.013131 | 0.011601 | 0.009524 | 0.013961 | 0.011924 | 0.150639 | 0.025150 | 0.004457 | 0.078428 | 0.018308 | 137 | 29.233465 | 0.000000 | 0.169200 | 0.222963 | 1500.000000 | 7.178900 | 10.752850 | 0.018308 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| trend | 0.011924 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| gbdt | 0.011924 |
| none | 0.011924 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| retrieval_only | 0.011924 |
| retrieval_ranker_fusion | 0.011924 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| gbdt | retrieval_ranker_fusion | 0.011924 |
| none | retrieval_only | 0.011924 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| trend | 0.018108 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.010996 | None | 0.017907 | None | 0.169200 | None | 0.000000 | None |
| none | 0.011240 | None | 0.018308 | None | 0.169200 | None | 0.000000 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| trend | none | 0.000000 |

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
