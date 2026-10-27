# KuaiRec Recall Diagnostics

This note records the valid rerun after fixing KuaiRec timestamp parsing. Earlier real-data numbers that showed `2/9872` retrieved positives are invalidated because numeric Unix-second timestamps were parsed as nanoseconds, collapsing processed timestamps to `1` and corrupting the temporal split.

## Valid Run Setup

- Raw data: `data/raw/kuairec_official/extracted/KuaiRec 2.0/data`
- Source matrix: `big_matrix.csv`
- Subset: `1000` users, `7483` items, `300000` interactions
- Flags: `is_sample_fallback=false`, `is_simulated=false`
- Positive label: `watch_ratio >= 1.0`
- Split: per-user temporal split after random interaction subsampling
- Candidate pool: `500`
- Retrievers: `popular,itemcf,bpr,lightgcn,two_tower,sasrec,content_faiss`

## Main Findings

- The framework bug was real: after the timestamp fix, test candidate positive recall improved from about `0.02%` to `9.88%`.
- Retrieval is still the bottleneck: best retrieval-only row is `sasrec` with `Recall@100=0.042851` and `Recall@10=0.003482`.
- Rankers do not rescue missing candidates. `din + retrieval_ranker_fusion` is effectively tied with `sasrec` retrieval-only, and `deepfm` is worse in this quick ablation.
- The remaining recall weakness is concentrated in cold/tail items, not new users. All test-positive users are warm-history users, but many target items have low or zero train positive counts.

## Candidate Audit

| split | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 78090 | 0 | 78090 | 0.000000 | 1.000000 |
| val | 9162 | 1493 | 7669 | 0.162956 | 0.837044 |
| test | 9287 | 918 | 8369 | 0.098848 | 0.901152 |
| overall | 96539 | 2411 | 94128 | 0.024974 | 0.975026 |

## Popularity Bucket Recall

| item popularity bucket | positives | retrieved | recall |
| --- | ---: | ---: | ---: |
| cold_0 | 2044 | 8 | 0.003914 |
| pop_1 | 505 | 14 | 0.027723 |
| pop_2_3 | 870 | 12 | 0.013793 |
| pop_4_10 | 2480 | 47 | 0.018952 |
| pop_11_30 | 2201 | 100 | 0.045434 |
| pop_31_plus | 1187 | 737 | 0.620893 |

## Retrieved Positive Source Counts

| source | retrieved positives in fused candidate pool |
| --- | ---: |
| popular | 662 |
| lightgcn | 654 |
| sasrec | 599 |
| two_tower | 580 |
| bpr | 480 |
| itemcf | 291 |
| content_faiss | 190 |

## Interpretation

The fixed framework no longer creates artificial all-cold temporal splits. However, the current recall stack is still biased toward head items. Head positives with train popularity `31+` are retrieved at about `62%`, while cold and low-popularity positives are mostly missed. ContentFaiss is not compensating enough because the official KuaiRec item side information in this path is mostly category/tag metadata rather than rich natural-language item text.

## Next Engineering Priorities

1. Add a tail-aware candidate branch: category co-occurrence, author/category trending, and recent-popularity retrieval should contribute candidates before RRF.
2. Evaluate per-retriever candidate recall directly in the report instead of showing only fusion-level candidate recall.
3. Tune SASRec/LightGCN with more epochs and validation-based model selection; current neural retrievers used a 1-epoch quick run.
4. Add tensor-cache or Arrow-batch loading for ranker samples; the current pandas `iloc` dataset is too slow for repeated ablations.
5. Treat injected positives as supervised audit rows only and keep reporting positive injection rate; do not claim ranker quality when injection remains high.
