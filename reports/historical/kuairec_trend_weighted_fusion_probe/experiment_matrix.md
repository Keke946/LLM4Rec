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
- candidate_pool_k: `500`
- cutoffs: `[5, 10, 20, 50, 100]`
- retriever list: `popular, trend, itemcf, bpr, lightgcn, two_tower, sasrec, content_faiss, metadata`
- ranker list: `none`
- RRF source weights: `{"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7}`

## Candidate Recall Before Ranker

| split | rows | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 578090 | 78090 | 0 | 78090 | 0.000000 | 1.000000 |
| val | 499669 | 9162 | 1493 | 7669 | 0.162956 | 0.837044 |
| test | 501869 | 9287 | 918 | 8369 | 0.098848 | 0.901152 |
| overall |  | 96539 | 2411 | 94128 | 0.024974 | 0.975026 |

## Top Runs

| dataset | retrievers | fusion | ranker | rerank_mode | business_reranker | candidate_pool_k | final_top_k | source_weights | candidate_positive_recall | positive_injection_rate | status | AUC | LogLoss | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank | HitRate@10 | HitRate@100 | HitRate@20 | HitRate@5 | HitRate@50 | MAP@10 | MRR@10 | NDCG@10 | NDCG@100 | NDCG@20 | NDCG@5 | NDCG@50 | Precision@10 | Precision@100 | Precision@20 | Precision@5 | Precision@50 | Recall@10 | Recall@100 | Recall@20 | Recall@5 | Recall@50 | CatalogCoverage | ItemCoverage | AveragePopularity | TailRatio | Diversity | Novelty | avg_candidates | latency_p50_ms | latency_p95_ms | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairec | fusion | RRF | none | retrieval_only | none | 500 | 10 | {"bpr": 0.7, "content_faiss": 0.5, "itemcf": 0.8, "lightgcn": 0.9, "metadata": 0.25, "popular": 0.4, "sasrec": 1.1, "trend": 1.6, "two_tower": 0.7} | 0.098848 | 0.901152 | ok | None | None | 0.004594 | None | 0.055058 | None | 0.189595 | None | 0.000132 | None | 0.046606 | 0.319149 | 0.087133 | 0.018237 | 0.171226 | 0.001263 | 0.010375 | 0.004594 | 0.018671 | 0.006690 | 0.003230 | 0.011070 | 0.004661 | 0.004103 | 0.004610 | 0.003647 | 0.003951 | 0.004951 | 0.043322 | 0.009823 | 0.001812 | 0.020420 | 0.055058 | 412 | 72.789990 | 0.000132 | 0.189595 | 0.164519 | 500.000000 | 184.821200 | 336.181060 | 0.055058 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| fusion | 0.004951 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| none | 0.004951 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| retrieval_only | 0.004951 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| none | retrieval_only | 0.004951 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| fusion | 0.055058 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.004594 | None | 0.055058 | None | 0.189595 | None | 0.000132 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| fusion | none | 0.000132 |

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
