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
- retriever list: `popular, itemcf, bpr, lightgcn, two_tower, sasrec, content_faiss`
- ranker list: `none`

## Candidate Recall Before Ranker

| split | rows | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 578090 | 78090 | 0 | 78090 | 0.000000 | 1.000000 |
| val | 499669 | 9162 | 1493 | 7669 | 0.162956 | 0.837044 |
| test | 501869 | 9287 | 918 | 8369 | 0.098848 | 0.901152 |
| overall |  | 96539 | 2411 | 94128 | 0.024974 | 0.975026 |

## Top Runs

| dataset | retrievers | fusion | ranker | rerank_mode | business_reranker | candidate_pool_k | final_top_k | candidate_positive_recall | positive_injection_rate | status | AUC | LogLoss | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank | HitRate@10 | HitRate@100 | HitRate@20 | HitRate@5 | HitRate@50 | MAP@10 | MRR@10 | NDCG@10 | NDCG@100 | NDCG@20 | NDCG@5 | NDCG@50 | Precision@10 | Precision@100 | Precision@20 | Precision@5 | Precision@50 | Recall@10 | Recall@100 | Recall@20 | Recall@5 | Recall@50 | CatalogCoverage | ItemCoverage | AveragePopularity | TailRatio | Diversity | Novelty | avg_candidates | latency_p50_ms | latency_p95_ms | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairec | sasrec | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.004462 | None | 0.060136 | None | 0.187953 | None | 0.032148 | None | 0.042553 | 0.314083 | 0.084093 | 0.021277 | 0.205674 | 0.001334 | 0.012368 | 0.004462 | 0.018735 | 0.006004 | 0.004190 | 0.012490 | 0.004357 | 0.004164 | 0.004559 | 0.004255 | 0.004883 | 0.003482 | 0.042851 | 0.007593 | 0.001653 | 0.023499 | 0.060136 | 450 | 40.923121 | 0.032148 | 0.187953 | 0.235726 | 497.883485 | 10.012000 | 13.908770 | 0.060136 |
| kuairec | fusion | RRF | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.002281 | None | 0.060404 | None | 0.198207 | None | 0.000142 | None | 0.024316 | 0.154002 | 0.049645 | 0.007092 | 0.095238 | 0.000593 | 0.004654 | 0.002281 | 0.008476 | 0.003440 | 0.001281 | 0.006022 | 0.002432 | 0.001773 | 0.002482 | 0.001418 | 0.002087 | 0.002727 | 0.018843 | 0.005191 | 0.000898 | 0.011857 | 0.060404 | 452 | 78.702796 | 0.000142 | 0.198207 | 0.159284 | 500.000000 | 93.208400 | 155.386510 | 0.060404 |
| kuairec | popular | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.001971 | None | 0.016972 | None | 0.203779 | None | 0.000000 | None | 0.020263 | 0.114488 | 0.043566 | 0.007092 | 0.068896 | 0.000503 | 0.003955 | 0.001971 | 0.006526 | 0.003345 | 0.001155 | 0.004633 | 0.002128 | 0.001317 | 0.002381 | 0.001418 | 0.001540 | 0.002327 | 0.013861 | 0.005325 | 0.000624 | 0.008287 | 0.016972 | 127 | 84.226353 | 0.000000 | 0.203779 | 0.156325 | 498.545086 | 3.818200 | 9.702930 | 0.016972 |
| kuairec | two_tower | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.002282 | None | 0.044367 | None | 0.195380 | None | 0.000000 | None | 0.018237 | 0.124620 | 0.037487 | 0.012158 | 0.073961 | 0.000791 | 0.005686 | 0.002282 | 0.007291 | 0.003173 | 0.002243 | 0.004952 | 0.001824 | 0.001408 | 0.001874 | 0.002432 | 0.001560 | 0.002191 | 0.015755 | 0.004023 | 0.001589 | 0.008547 | 0.044367 | 332 | 69.115329 | 0.000000 | 0.195380 | 0.164554 | 498.686930 | 8.519200 | 11.623570 | 0.044367 |
| kuairec | itemcf | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.001722 | None | 0.616197 | None | 0.210689 | None | 0.409878 | None | 0.016211 | 0.134752 | 0.031408 | 0.005066 | 0.071935 | 0.000530 | 0.003992 | 0.001722 | 0.006887 | 0.002460 | 0.001130 | 0.004329 | 0.001621 | 0.001489 | 0.001570 | 0.001013 | 0.001540 | 0.001812 | 0.015680 | 0.003573 | 0.000583 | 0.007988 | 0.616197 | 4611 | 38.155177 | 0.409878 | 0.210689 | 0.359205 | 499.001013 | 12.568300 | 23.606510 | 0.616197 |
| kuairec | lightgcn | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.001243 | None | 0.016571 | None | 0.187386 | None | 0.000000 | None | 0.014184 | 0.170213 | 0.032421 | 0.005066 | 0.068896 | 0.000290 | 0.002333 | 0.001243 | 0.008732 | 0.002361 | 0.000734 | 0.004309 | 0.001418 | 0.001925 | 0.001722 | 0.001013 | 0.001540 | 0.001497 | 0.022854 | 0.004030 | 0.000591 | 0.008781 | 0.016571 | 124 | 78.594164 | 0.000000 | 0.187386 | 0.158857 | 498.602837 | 48.925200 | 62.526270 | 0.016571 |
| kuairec | bpr | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.001609 | None | 0.352532 | None | 0.190557 | None | 0.003060 | None | 0.014184 | 0.134752 | 0.031408 | 0.008105 | 0.073961 | 0.000516 | 0.004404 | 0.001609 | 0.006686 | 0.002405 | 0.001595 | 0.004060 | 0.001418 | 0.001489 | 0.001570 | 0.001621 | 0.001540 | 0.001289 | 0.014986 | 0.003359 | 0.000815 | 0.007063 | 0.352532 | 2638 | 57.773799 | 0.003060 | 0.190557 | 0.179215 | 498.806484 | 6.460900 | 9.989130 | 0.352532 |
| kuairec | content_faiss | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | None | None | 0.000663 | None | 0.144327 | None | 0.216991 | None | 0.105147 | None | 0.007092 | 0.142857 | 0.023303 | 0.004053 | 0.069909 | 0.000176 | 0.001652 | 0.000663 | 0.006377 | 0.001439 | 0.000670 | 0.003356 | 0.000709 | 0.001581 | 0.001165 | 0.000811 | 0.001459 | 0.000579 | 0.015590 | 0.002152 | 0.000364 | 0.006581 | 0.144327 | 1080 | 14.497001 | 0.105147 | 0.216991 | 0.494480 | 499.075988 | 4.945500 | 8.412790 | 0.144327 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| sasrec | 0.003482 |
| fusion | 0.002727 |
| popular | 0.002327 |
| two_tower | 0.002191 |
| itemcf | 0.001812 |
| lightgcn | 0.001497 |
| bpr | 0.001289 |
| content_faiss | 0.000579 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| none | 0.001988 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| retrieval_only | 0.001988 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| none | retrieval_only | 0.001988 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| itemcf | 0.616197 |
| bpr | 0.352532 |
| content_faiss | 0.144327 |
| fusion | 0.060404 |
| sasrec | 0.060136 |
| two_tower | 0.044367 |
| popular | 0.016972 |
| lightgcn | 0.016571 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.001971 | None | 0.016972 | None | 0.203779 | None | 0.000000 | None |
| none | 0.001722 | None | 0.616197 | None | 0.210689 | None | 0.409878 | None |
| none | 0.001609 | None | 0.352532 | None | 0.190557 | None | 0.003060 | None |
| none | 0.001243 | None | 0.016571 | None | 0.187386 | None | 0.000000 | None |
| none | 0.002282 | None | 0.044367 | None | 0.195380 | None | 0.000000 | None |
| none | 0.004462 | None | 0.060136 | None | 0.187953 | None | 0.032148 | None |
| none | 0.000663 | None | 0.144327 | None | 0.216991 | None | 0.105147 | None |
| none | 0.002281 | None | 0.060404 | None | 0.198207 | None | 0.000142 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| itemcf | none | 0.409878 |
| content_faiss | none | 0.105147 |
| sasrec | none | 0.032148 |
| bpr | none | 0.003060 |
| fusion | none | 0.000142 |
| lightgcn | none | 0.000000 |
| popular | none | 0.000000 |
| two_tower | none | 0.000000 |

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
