# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 493000.000000 |
| positives | 9287.000000 |
| retrieved_positives | 4496.000000 |
| injected_positives | 4791.000000 |
| candidate_positive_recall | 0.484118 |
| positive_injection_rate | 0.515882 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 0 | 0.000000 |
| pop_1 | 505 | 0 | 0.000000 |
| pop_2_3 | 870 | 119 | 0.136782 |
| pop_4_10 | 2480 | 1864 | 0.751613 |
| pop_11_30 | 2201 | 1654 | 0.751477 |
| pop_31_plus | 1187 | 859 | 0.723673 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 11 | 0.500000 |
| h21_50 | 9265 | 4485 | 0.484080 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| trend | 488209 | 4496 | 0.484118 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
