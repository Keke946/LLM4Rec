# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 1482954.000000 |
| positives | 9287.000000 |
| retrieved_positives | 6833.000000 |
| injected_positives | 2454.000000 |
| candidate_positive_recall | 0.735760 |
| positive_injection_rate | 0.264240 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 0 | 0.000000 |
| pop_1 | 505 | 469 | 0.928713 |
| pop_2_3 | 870 | 831 | 0.955172 |
| pop_4_10 | 2480 | 2406 | 0.970161 |
| pop_11_30 | 2201 | 2049 | 0.930940 |
| pop_31_plus | 1187 | 1078 | 0.908172 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 18 | 0.818182 |
| h21_50 | 9265 | 6815 | 0.735564 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| trend | 1480500 | 6833 | 0.735760 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
