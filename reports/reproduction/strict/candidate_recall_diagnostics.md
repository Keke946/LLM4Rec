# Candidate Recall Diagnostics: kuairec

Split: `test`

## Summary

| metric | value |
| --- | --- |
| rows | 1395247.000000 |
| positives | 8952.000000 |
| retrieved_positives | 5705.000000 |
| injected_positives | 3247.000000 |
| candidate_positive_recall | 0.637288 |
| positive_injection_rate | 0.362712 |
| split_positive_item_in_train_ratio | 0.672537 |
| split_unique_positive_items | 1437.000000 |

## By Item Popularity

| item_popularity_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| cold_0 | 3000 | 0 | 0.000000 |
| pop_1 | 407 | 372 | 0.914005 |
| pop_2_3 | 898 | 880 | 0.979955 |
| pop_4_10 | 2103 | 2039 | 0.969567 |
| pop_11_30 | 1516 | 1438 | 0.948549 |
| pop_31_plus | 1028 | 976 | 0.949416 |

## By History Length

| history_len_bucket | positives | retrieved_positives | candidate_positive_recall |
| --- | --- | --- | --- |
| h0 | 0 | 0 | nan |
| h1_5 | 0 | 0 | nan |
| h6_20 | 22 | 19 | 0.863636 |
| h21_50 | 8930 | 5686 | 0.636730 |
| h50_plus | 0 | 0 | nan |

## By Retrieval Source

| source | candidate_rows | retrieved_positive_hits | credited_positive_recall |
| --- | --- | --- | --- |
| trend | 1392000 | 5705 | 0.637288 |

## Reading Guide

`candidate_positive_recall` measures how many split positives were retrieved before audited positive injection. High `positive_injection_rate` means the ranker is mostly learning from positives that the recall stage failed to surface. Source rows are credited when a retrieved positive appears in that source's fused candidate metadata.
