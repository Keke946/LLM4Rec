# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 990091.000000 |
| positives | 9287.000000 |
| retrieved_positives | 6196.000000 |
| injected_positives | 3091.000000 |
| candidate_positive_recall | 0.667169 |
| positive_injection_rate | 0.332831 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 0 | 0.000000 |
| pop_1 | 505 | 206 | 0.407921 |
| pop_2_3 | 870 | 769 | 0.883908 |
| pop_4_10 | 2480 | 2277 | 0.918145 |
| pop_11_30 | 2201 | 1940 | 0.881418 |
| pop_31_plus | 1187 | 1004 | 0.845830 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 17 | 0.772727 |
| h21_50 | 9265 | 6179 | 0.666919 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| trend | 987000 | 6196 | 0.667169 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
