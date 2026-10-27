# Final Experiment Summary V2

## 1. Project Goal

This project is a KuaiRec short-video multi-stage recommendation experiment. The goal is to move the repo from a demo-style prototype toward a real-data recommendation algorithm project with retriever recall, candidate fusion, temporal ranker training, reranking, evaluation, and diagnostics. LLM components are not used to directly choose recommendation item IDs.

## 2. Dataset and Split

- dataset: `kuairec`
- users/items/interactions: `1000` / `7483` / `300000`
- positive label definition: KuaiRec positive labels follow the processed adapter, typically `watch_ratio >= 1.0`.
- temporal split: per-user timestamp split into train/val/test.
- seen filter: `train,val`
- candidate pool: RRF over configured retrievers, candidate_pool_k=`1500`, cutoffs=`[5, 10, 20, 50, 100]`.

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

| experiment | status | Recall@10 | NDCG@10 | MRR@10 | Recall@100 | candidate_positive_recall |
| --- | --- | --- | --- | --- | --- | --- |
| retrieval_only | ok | 0.012151 | 0.011062 | 0.023785 | 0.151658 | 0.735760 |
| gbdt_full | ok | 0.012672 | 0.015193 | 0.040112 | 0.125811 | 0.735760 |
| gbdt_top50 | ok | 0.026115 | 0.029657 | 0.079903 | 0.151658 | 0.735760 |
| gbdt_top100 | ok | 0.022871 | 0.025955 | 0.067142 | 0.151658 | 0.735760 |
| lambdarank_full | ok | 0.008716 | 0.009050 | 0.022420 | 0.084576 | 0.735760 |
| lambdarank_top50 | ok | 0.017210 | 0.017448 | 0.041612 | 0.151658 | 0.735760 |
| lambdarank_top100 | ok | 0.014470 | 0.014879 | 0.036811 | 0.151658 | 0.735760 |

## 7. Negative Sampling Ablation

| negative_strategy | status | Recall@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| random_negative | ok | 0.024111 | 0.028056 | 0.075194 |
| exposed_low_watch_negative | ok | 0.023132 | 0.021952 | 0.048505 |
| retrieved_hard_negative | ok | 0.009148 | 0.008613 | 0.019666 |
| mixed_negative | ok | 0.024624 | 0.025097 | 0.062091 |

## 8. Rerank Scope Ablation

The matrix includes full ranker-only rerank and `ranker_top_50` / `ranker_top_100` head rerank. Head rerank can be stronger at this stage because it preserves retrieval recall depth while letting the ranker reorder the most plausible head candidates.

## 9. Failure Case Analysis

See `reports\kuairec_ltr_ablation\failure_cases.csv` and `reports\kuairec_ltr_ablation\failure_case_summary.md`. Failure reasons distinguish `not_retrieved`, `retrieved_but_low_rank`, `cold_item`, `tail_item`, `popularity_bias`, `category_mismatch`, and `insufficient_user_history`.

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
{
  "positives": 32620,
  "retrieved_positives": 22269,
  "injected_positives": 10351,
  "positive_injection_rate": 0.3173206621704476,
  "candidate_positive_recall": 0.6826793378295524
}
```
