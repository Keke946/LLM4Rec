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
- ranker list: `none, deepfm, din`

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
| kuairec | sasrec | single | none | retrieval_only | none | 500 | 10 | 0.098848 | 0.901152 | ok |  |  | 0.004462 | None | 0.060136 | None | 0.187953 | None | 0.032148 | None | 0.042553 | 0.314083 | 0.084093 | 0.021277 | 0.205674 | 0.001334 | 0.012368 | 0.004462 | 0.018735 | 0.006004 | 0.004190 | 0.012490 | 0.004357 | 0.004164 | 0.004559 | 0.004255 | 0.004883 | 0.003482 | 0.042851 | 0.007593 | 0.001653 | 0.023499 | 0.060136 | 450 | 40.923121 | 0.032148 | 0.187953 | 0.235726 | 497.883485 | 11.513300 | 17.586390 | 0.060136 |
| kuairec | sasrec | single | din | retrieval_ranker_fusion | none | 500 | 10 | 0.098848 | 0.901152 | ok | 0.839434 | 0.197443 | 0.004462 | None | 0.060270 | None | 0.187305 | None | 0.032432 | None | 0.042553 | 0.313070 | 0.084093 | 0.021277 | 0.205674 | 0.001334 | 0.012368 | 0.004462 | 0.018729 | 0.006004 | 0.004190 | 0.012453 | 0.004357 | 0.004164 | 0.004559 | 0.004255 | 0.004863 | 0.003482 | 0.042871 | 0.007593 | 0.001653 | 0.023414 | 0.060270 | 451 | 40.653789 | 0.032432 | 0.187305 | 0.237316 | 497.883485 | 9.819500 | 14.432490 | 0.060270 |
| kuairec | sasrec | single | deepfm | retrieval_ranker_fusion | none | 500 | 10 | 0.098848 | 0.901152 | ok | 0.546772 | 2.343739 | 0.004637 | None | 0.124282 | None | 0.183495 | None | 0.036353 | None | 0.040527 | 0.287741 | 0.073961 | 0.024316 | 0.176292 | 0.001481 | 0.013543 | 0.004637 | 0.017064 | 0.005523 | 0.004981 | 0.010742 | 0.004255 | 0.003840 | 0.003799 | 0.004863 | 0.004053 | 0.003307 | 0.037438 | 0.006440 | 0.001750 | 0.019005 | 0.124282 | 930 | 38.661277 | 0.036353 | 0.183495 | 0.239078 | 497.883485 | 10.550200 | 19.123040 | 0.124282 |
| kuairec | sasrec | single | din | ranker_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | 0.839434 | 0.197443 | 0.003571 | None | 0.134973 | None | 0.214610 | None | 0.011884 | None | 0.030395 | 0.246201 | 0.047619 | 0.014184 | 0.133739 | 0.001218 | 0.008932 | 0.003571 | 0.013587 | 0.004110 | 0.002956 | 0.008326 | 0.003242 | 0.002999 | 0.002584 | 0.002837 | 0.002958 | 0.002968 | 0.030835 | 0.004734 | 0.001617 | 0.015271 | 0.134973 | 1010 | 22.292806 | 0.011884 | 0.214610 | 0.324981 | 497.883485 | 9.972500 | 14.095270 | 0.134973 |
| kuairec | sasrec | single | deepfm | ranker_only | none | 500 | 10 | 0.098848 | 0.901152 | ok | 0.546772 | 2.343739 | 0.002410 | None | 0.156354 | None | 0.152837 | None | 0.036900 | None | 0.023303 | 0.240122 | 0.049645 | 0.009119 | 0.134752 | 0.000702 | 0.005963 | 0.002410 | 0.012491 | 0.003653 | 0.001786 | 0.007838 | 0.002330 | 0.002766 | 0.002533 | 0.001824 | 0.002837 | 0.002367 | 0.030008 | 0.005391 | 0.000853 | 0.016333 | 0.156354 | 1170 | 31.865157 | 0.036900 | 0.152837 | 0.264845 | 497.883485 | 10.391500 | 17.821060 | 0.156354 |

## Retriever Comparison

| retrievers | Recall@10 |
| --- | --- |
| sasrec | 0.003121 |

## Ranker Comparison

| ranker | Recall@10 |
| --- | --- |
| none | 0.003482 |
| din | 0.003225 |
| deepfm | 0.002837 |

## Rerank Fusion Comparison

| rerank_mode | Recall@10 |
| --- | --- |
| retrieval_only | 0.003482 |
| retrieval_ranker_fusion | 0.003394 |
| ranker_only | 0.002667 |

## Feature Ablation

The matrix keeps `retrieval_only`, `ranker_only`, and `retrieval_ranker_fusion` rows. Retrieval-side features include retrieval score, retrieval source, source rank, source count, item popularity, user activity level, and history length.

| ranker | rerank_mode | Recall@10 |
| --- | --- | --- |
| none | retrieval_only | 0.003482 |
| din | retrieval_ranker_fusion | 0.003482 |
| deepfm | retrieval_ranker_fusion | 0.003307 |
| din | ranker_only | 0.002968 |
| deepfm | ranker_only | 0.002367 |

## Retrieval Source Contribution

Single-source rows estimate marginal source quality. The `fusion` row uses RRF over configured retrievers.

| retrievers | Coverage |
| --- | --- |
| sasrec | 0.107203 |

## Business Rerank Before / After

| business_reranker | NDCG@10_before_business_rerank | NDCG@10_after_business_rerank | Coverage_before_business_rerank | Coverage_after_business_rerank | Diversity_before_business_rerank | Diversity_after_business_rerank | TailRatio_before_business_rerank | TailRatio_after_business_rerank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.004462 | None | 0.060136 | None | 0.187953 | None | 0.032148 | None |
| none | 0.002410 | None | 0.156354 | None | 0.152837 | None | 0.036900 | None |
| none | 0.004637 | None | 0.124282 | None | 0.183495 | None | 0.036353 | None |
| none | 0.003571 | None | 0.134973 | None | 0.214610 | None | 0.011884 | None |
| none | 0.004462 | None | 0.060270 | None | 0.187305 | None | 0.032432 | None |

## Cold-Start Analysis

Cold-start is approximated through user/item coverage and failure cases in `failure_cases.json`. A production version should add explicit buckets for new users, short-history users, new items, and tail items.

## Tail Item Analysis

| retrievers | business_reranker | TailRatio |
| --- | --- | --- |
| sasrec | none | 0.029943 |

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
