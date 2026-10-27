# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 1482785.000000 |
| positives | 9287.000000 |
| retrieved_positives | 7002.000000 |
| injected_positives | 2285.000000 |
| candidate_positive_recall | 0.753957 |
| positive_injection_rate | 0.246043 |
| split_positive_item_in_train_ratio | 0.784394 |
| split_unique_positive_items | 1836.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 165 | 0.080724 |
| pop_1 | 505 | 476 | 0.942574 |
| pop_2_3 | 870 | 838 | 0.963218 |
| pop_4_10 | 2480 | 2427 | 0.978629 |
| pop_11_30 | 2201 | 2058 | 0.935030 |
| pop_31_plus | 1187 | 1038 | 0.874473 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 18 | 0.818182 |
| h21_50 | 9265 | 6984 | 0.753805 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| exposure | 1306786 | 6917 | 0.744805 |
| trend | 1407348 | 6794 | 0.731560 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
