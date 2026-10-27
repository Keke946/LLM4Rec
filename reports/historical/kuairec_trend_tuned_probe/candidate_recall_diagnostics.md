# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 498166.000000 |
| positives | 9287.000000 |
| retrieved_positives | 4621.000000 |
| injected_positives | 4666.000000 |
| candidate_positive_recall | 0.497577 |
| positive_injection_rate | 0.502423 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 0 | 0.000000 |
| pop_1 | 505 | 0 | 0.000000 |
| pop_2_3 | 870 | 245 | 0.281609 |
| pop_4_10 | 2480 | 1876 | 0.756452 |
| pop_11_30 | 2201 | 1663 | 0.755566 |
| pop_31_plus | 1187 | 837 | 0.705139 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 12 | 0.545455 |
| h21_50 | 9265 | 4609 | 0.497464 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| trend | 493500 | 4621 | 0.497577 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
