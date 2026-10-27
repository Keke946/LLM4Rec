# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 501869.000000 |
| positives | 9287.000000 |
| retrieved_positives | 918.000000 |
| injected_positives | 8369.000000 |
| candidate_positive_recall | 0.098848 |
| positive_injection_rate | 0.901152 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 8 | 0.003914 |
| pop_1 | 505 | 14 | 0.027723 |
| pop_2_3 | 870 | 12 | 0.013793 |
| pop_4_10 | 2480 | 47 | 0.018952 |
| pop_11_30 | 2201 | 100 | 0.045434 |
| pop_31_plus | 1187 | 737 | 0.620893 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 2 | 0.090909 |
| h21_50 | 9265 | 916 | 0.098867 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| popular | 362027 | 662 | 0.071282 |
| lightgcn | 371522 | 654 | 0.070421 |
| sasrec | 205052 | 599 | 0.064499 |
| two_tower | 359323 | 580 | 0.062453 |
| bpr | 277499 | 480 | 0.051685 |
| itemcf | 175343 | 291 | 0.031334 |
| content_faiss | 98735 | 190 | 0.020459 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
