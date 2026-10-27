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
- candidate_pool_k: `1000`
- cutoffs: `[5, 10, 20, 50, 100]`
- retriever list: `popular, trend, itemcf, bpr, lightgcn, two_tower, sasrec, content_faiss, metadata`
- ranker list: `none, gbdt`
- RRF source weights: `{"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7}`

## Candidate Recall Before Ranker

| split | rows | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1078090 | 78090 | 0 | 78090 | 0.000000 | 1.000000 |
| val | 986058 | 9162 | 7104 | 2058 | 0.775377 | 0.224623 |
| test | 990091 | 9287 | 6196 | 3091 | 0.667169 | 0.332831 |
| overall |  | 96539 | 13300 | 83239 | 0.137768 | 0.862232 |

## Top Runs

| dataset | retrievers | fusion | ranker | rerank_mode | business_reranker | candidate_pool_k | final_top_k | source_weights | candidate_positive_recall | positive_injection_rate | status | AUC | LogLoss | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank | HitRate@10 | HitRate@100 | HitRate@20 | HitRate@5 | HitRate@50 | MAP@10 | MRR@10 | NDCG@10 | NDCG@100 | NDCG@20 | NDCG@5 | NDCG@50 | Precision@10 | Precision@100 | Precision@20 | Precision@5 | Precision@50 | Recall@10 | Recall@100 | Recall@20 | Recall@5 | Recall@50 | CatalogCoverage | ItemCoverage | AveragePopularity | TailRatio | Diversity | Novelty | avg_candidates | latency_p50_ms | latency_p95_ms | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairec | trend | single | gbdt | ranker_only | none | 1000 | 10 | {"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7} | 0.667169 | 0.332831 | ok | 0.683370 | 0.586422 | 0.025443 | None | 0.059201 | None | 0.181287 | None | 0.006302 | None | 0.169200 | 0.714286 | 0.288754 | 0.111449 | 0.518744 | 0.010097 | 0.073458 | 0.025443 | 0.075329 | 0.032335 | 0.028004 | 0.051583 | 0.018136 | 0.014367 | 0.017275 | 0.023708 | 0.015866 | 0.020090 | 0.159580 | 0.038461 | 0.013702 | 0.086503 | 0.059201 | 443 | 20.170598 | 0.006302 | 0.181287 | 0.251049 | 1000.000000 | 5.486900 | 7.744800 | 0.059201 |
| kuairec | trend | single | none | retrieval_only | none | 1000 | 10 | {"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7} | 0.667169 | 0.332831 | ok |  |  | 0.010996 | None | 0.017907 | None | 0.169200 | None | 0.000000 | None | 0.102330 | 0.688956 | 0.192503 | 0.045593 | 0.483283 | 0.003112 | 0.023785 | 0.010996 | 0.060328 | 0.016457 | 0.008027 | 0.037443 | 0.011246 | 0.013040 | 0.011601 | 0.009321 | 0.013921 | 0.011924 | 0.149920 | 0.025259 | 0.004330 | 0.078487 | 0.017907 | 134 | 30.127042 | 0.000000 | 0.169200 | 0.220653 | 1000.000000 | 3.766000 | 6.614090 | 0.017907 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| trend | 0.016007 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| gbdt | 0.020090 |
| none | 0.011924 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| ranker_only | 0.020090 |
| retrieval_only | 0.011924 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| gbdt | ranker_only | 0.020090 |
| none | retrieval_only | 0.011924 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| trend | 0.038554 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.010996 | None | 0.017907 | None | 0.169200 | None | 0.000000 | None |
| none | 0.025443 | None | 0.059201 | None | 0.181287 | None | 0.006302 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| trend | none | 0.003151 |

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
