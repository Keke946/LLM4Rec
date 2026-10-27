# LLM4Rec Final Experiment Summary

## 1. Environment Verification
- Python version: `Python 3.11.9`
- compileall result: PASS
- unittest result: PASS (18 tests OK)
- secret scan result: PASS
- make availability: NOT AVAILABLE on this Windows shell; manual pipeline was used.

## 2. Completed Experiments
- MovieLens smoke experiment: completed. Reports were generated under `reports/movielens`.
- KuaiRec official experiment: completed on official KuaiRec 2.0 raw data under `data/raw/kuairec_official/extracted/KuaiRec 2.0/data`.
- Earlier safe-failure check: PASS. `data/raw/kuairec` contains `simulated_data_manifest.json`, so it must not be treated as official real-data input.
- Current valid real-data run: PASS after fixing KuaiRec timestamp parsing and rerunning prepare -> build_features -> build_ranker_candidates -> train rankers -> run matrix.

## 3. Dataset Statistics

### MovieLens Smoke
- num_users: 4
- num_items: 6
- num_interactions: 16
- positive rate: 0.5625
- split_stats: `{'test': 4, 'train': 8, 'val': 4}`
- is_sample_fallback: True
- is_simulated: False
- data_origin: `D:\qqfiles\1_LLM4Rec-main\LLM4Rec-main\examples\sample_data\movielens`

MovieLens is a smoke test only. It used the repository sample fallback because `data/raw/movielens` was absent.

Important metric note: `NDCG@10`, `Recall@100`, and `MRR@10` were unavailable/undefined in the MovieLens smoke output. The tiny fallback dataset plus strict train/val seen filtering produced no evaluable recommendation lists for those ranking metrics. This is not a real ranking-quality result.

## 4. Candidate-Level Ranker Sample Audit
- candidate_pool_k: 200
- overall candidate_positive_recall: 0.4444444444444444
- overall positive_injection_rate: 0.5555555555555556
- injected positives are audit-only: yes. `is_injected_positive` and `is_retrieved_positive` are audit fields; injected positives are not exposed as a `label_positive` source embedding.

Split-level audit:

| split | rows | positives | retrieved_positives | injected_positives | positive_injection_rate | candidate_positive_recall | negatives | users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 24 | 5 | 0 | 5 | 1.000000 | 0.000000 | 19 | 4 |
| val | 19 | 4 | 4 | 0 | 0.000000 | 1.000000 | 15 | 4 |
| test | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0 |

## 5. Retriever Comparison
MovieLens metrics rows: 240. Status counts: `{'ok': 240}`. Failed/missing/skipped rows: 0.

Best rows requested by metric:
- NDCG@10: unavailable/undefined in this smoke run.
- Recall@100: unavailable/undefined in this smoke run.
- MRR@10: unavailable/undefined in this smoke run.

Retriever-group summary:

| retrievers | rows | ok_rows | Coverage | CatalogCoverage | ItemCoverage | Diversity | TailRatio | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bpr | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| content_faiss | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fusion | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| itemcf | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| lightgcn | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| popular | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| sasrec | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| two_tower | 30 | 30 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Covered retrievers in the matrix: bpr, content_faiss, fusion, itemcf, lightgcn, popular, sasrec, two_tower.

## 6. Ranker Comparison
Ranker-group summary:

| ranker | rows | ok_rows | Coverage | CatalogCoverage | ItemCoverage | Diversity | TailRatio | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepfm | 72 | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| din | 72 | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| embedding_mlp | 72 | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| none | 24 | 24 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Observed rankers: deepfm, din, embedding_mlp, none. Because ranking metrics were unavailable in this tiny smoke output, no honest claim can be made that EmbeddingMLP, DeepFM, or DIN improved ranking quality here.

## 7. Retrieval-Ranker Fusion
Mode-group summary:

| rerank_mode | rows | ok_rows | Coverage | CatalogCoverage | ItemCoverage | Diversity | TailRatio | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranker_only | 72 | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| retrieval_only | 96 | 96 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| retrieval_ranker_fusion | 72 | 72 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Best retrieval-only run by NDCG@10: NDCG@10: unavailable/undefined in this smoke run.

Best retrieval-ranker-fusion run by NDCG@10: NDCG@10: unavailable/undefined in this smoke run.

Fusion conclusion: unavailable for quality comparison in this smoke run because ranking metrics are undefined. Do not claim fusion improves retrieval based on this MovieLens sample run.

## 8. Business Rerank
Business-reranker summary:

| business_reranker | rows | ok_rows | Coverage | CatalogCoverage | ItemCoverage | Diversity | TailRatio | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diversity | 80 | 80 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| diversity_tail | 80 | 80 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| none | 80 | 80 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Business rerank conclusion: the matrix includes `none`, `diversity`, and `diversity_tail`, but ranking tradeoffs cannot be claimed from this tiny smoke output. Coverage/Diversity/TailRatio columns are present for report-shape verification, yet the values are not sufficient evidence of product impact.

## 9. Failure Cases
Representative failure cases from `reports/movielens/failure_cases.json`:

No representative failure cases were recorded.

## 10. LLM Role and Safety
- LLM does not generate item IDs.
- LLM is used only for query rewrite/explanation/tool-interface style behavior.
- Hallucination guardrail is enabled through item-ID rejection checks.
- `python scripts/check_secrets.py` passed; no API key was used for these experiments.

## 11. Honest Limitations
- MovieLens is a smoke test, not proof of video recommendation quality.
- The MovieLens run used sample fallback data and should not be presented as a real-data performance result.
- Official KuaiRec raw data is now available under `data/raw/kuairec_official/extracted/KuaiRec 2.0/data`.
- `data/raw/kuairec` still contains simulated raw data; use `data/raw/kuairec_official/...` for real-data runs.
- A real KuaiRec subset has been run, but the observed offline ranking metrics are very low because candidate recall is dominated by temporal cold-item effects.
- Do not claim online CTR/watch-time lift or ranker improvement from this run.

## 12. KuaiRec Simulated Quick Run
- Status: completed as a simulated-data quick run, not official KuaiRec.
- Raw data marker: `data/raw/kuairec/simulated_data_manifest.json`
- Processed data flags: `is_sample_fallback=false`, `is_simulated=true`
- Subset used: `500` users, `5000` items, `50000` interactions.
- Positive rate: `0.64026`
- Pipeline: prepare -> build_features -> build_ranker_candidates -> train rankers -> run experiment matrix.
- Quick-run settings: `candidate_pool_k=200`, cutoffs `[5, 10, 20, 50, 100]`, final_top_k `10`, retriever training `1` epoch, ranker training `1` epoch.
- Retriever coverage in matrix: `popular`, `itemcf`, `bpr`, `lightgcn`, `two_tower`, `sasrec`, `content_faiss`, `fusion`.
- Rankers in matrix: `none`, `embedding_mlp`, `deepfm`, `din`.
- Matrix rows: `32`, all status `ok`.
- Reports: `reports/kuairec/metrics_summary.csv`, `reports/kuairec/experiment_report.md`, `reports/kuairec/experiment_matrix.md`, `reports/kuairec/failure_cases.json`.

Candidate audit:

| split | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- |
| train | 25594 | 0 | 25594 | 0.000000 | 1.000000 |
| val | 3180 | 151 | 3029 | 0.047484 | 0.952516 |
| test | 3239 | 53 | 3186 | 0.016363 | 0.983637 |
| overall | 32013 | 204 | 31809 | 0.006372 | 0.993628 |

Best observed recommendation rows:

| criterion | retriever | ranker | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | Coverage | TailRatio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best NDCG@10 | bpr | deepfm | 0.001025 | 0.000841 | 0.013663 | 0.003000 | 0.962400 | 0.239320 |
| best Recall@100 | content_faiss | embedding_mlp | 0.000000 | 0.000000 | 0.020865 | 0.000000 | 0.036000 | 0.208140 |

Ranker validation metrics:

| ranker | val_auc | val_logloss |
| --- | --- | --- |
| embedding_mlp | 0.956755 | 0.124303 |
| deepfm | 0.679707 | 3.545552 |
| din | 0.961872 | 0.094018 |

Interpretation:
- The end-to-end ranking metrics are extremely low. This is mainly a recall-stage issue: test candidate_positive_recall is only `0.016363`, and test positive_injection_rate is `0.983637`.
- The model can fit injected candidate-level labels on validation samples, but the retrievers are not surfacing enough held-out positives for strong offline Recall/NDCG.
- This run is useful for checking pipeline correctness and report generation, but it should not be sent out as official KuaiRec performance.
- No external API key was used.

## 13. KuaiRec Official Real-Data Run
- Status: completed on official KuaiRec 2.0 raw data downloaded from Zenodo.
- Raw data path: `data/raw/kuairec_official/extracted/KuaiRec 2.0/data`
- Source matrix: `big_matrix.csv`
- Processed data flags: `is_sample_fallback=false`, `is_simulated=false`
- Subset used: `1000` active users, `7483` items, `300000` interactions.
- Positive label: `watch_ratio >= 1.0`
- Positive rate: `0.32984`
- Split strategy: per-user temporal split after random subset sampling, with fixed Unix-second timestamp parsing.
- Quick-run settings: `candidate_pool_k=500`, cutoffs `[5, 10, 20, 50, 100]`, final_top_k `10`, retriever training `1` epoch, ranker quick ablation `1` epoch with explicit sampled-row metadata.
- Code upgrades made for this run: `--kuairec_matrix big`, `--max_interactions_strategy random`, KuaiRec `feat` category parsing, version-safe timestamp parsing, ndarray-safe diversity metric, cached ContentFaiss user profile embeddings, and reproducible quick caps for ranker ablations.
- Main retrieval report: `reports/kuairec/metrics_summary.csv`
- Ranker ablation report: `reports/kuairec_ranker_ablation/metrics_summary.csv`
- Recall diagnostics: `reports/kuairec/diagnostics.md`

Invalidated earlier note:
- The earlier real-data result with `test retrieved_positives=2/9872` and best `Recall@100=0.012110` is invalid.
- Root cause: numeric Unix-second KuaiRec timestamps were parsed by pandas as nanoseconds, so processed timestamps collapsed to `1`. The temporal split was therefore corrupted and created artificial cold-item drift.

Candidate audit:

| split | positives | retrieved_positives | injected_positives | candidate_positive_recall | positive_injection_rate |
| --- | --- | --- | --- | --- | --- |
| train | 78090 | 0 | 78090 | 0.000000 | 1.000000 |
| val | 9162 | 1493 | 7669 | 0.162956 | 0.837044 |
| test | 9287 | 918 | 8369 | 0.098848 | 0.901152 |
| overall | 96539 | 2411 | 94128 | 0.024974 | 0.975026 |

Retriever comparison:

| retriever | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | CatalogCoverage |
| --- | --- | --- | --- | --- | --- |
| sasrec | 0.004462 | 0.003482 | 0.042851 | 0.012368 | 0.060136 |
| lightgcn | 0.001243 | 0.001497 | 0.022854 | 0.002333 | 0.016571 |
| fusion | 0.002281 | 0.002727 | 0.018843 | 0.004654 | 0.060404 |
| two_tower | 0.002282 | 0.002191 | 0.015755 | 0.005686 | 0.044367 |
| itemcf | 0.001722 | 0.001812 | 0.015680 | 0.003992 | 0.616197 |
| content_faiss | 0.000663 | 0.000579 | 0.015590 | 0.001652 | 0.144327 |
| bpr | 0.001609 | 0.001289 | 0.014986 | 0.004404 | 0.352532 |
| popular | 0.001971 | 0.002327 | 0.013861 | 0.003955 | 0.016972 |

SASRec ranker ablation:

| ranker | rerank_mode | NDCG@10 | Recall@10 | Recall@100 | MRR@10 | AUC | LogLoss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | retrieval_only | 0.004462 | 0.003482 | 0.042851 | 0.012368 |  |  |
| deepfm | ranker_only | 0.002410 | 0.002367 | 0.030008 | 0.005963 | 0.546772 | 2.343739 |
| deepfm | retrieval_ranker_fusion | 0.004637 | 0.003307 | 0.037438 | 0.013543 | 0.546772 | 2.343739 |
| din | ranker_only | 0.003571 | 0.002968 | 0.030835 | 0.008932 | 0.839434 | 0.197443 |
| din | retrieval_ranker_fusion | 0.004462 | 0.003482 | 0.042871 | 0.012368 | 0.839434 | 0.197443 |

Tail/cold recall diagnosis:

| item popularity bucket | positives | retrieved | recall |
| --- | --- | --- | --- |
| cold_0 | 2044 | 8 | 0.003914 |
| pop_1 | 505 | 14 | 0.027723 |
| pop_2_3 | 870 | 12 | 0.013793 |
| pop_4_10 | 2480 | 47 | 0.018952 |
| pop_11_30 | 2201 | 100 | 0.045434 |
| pop_31_plus | 1187 | 737 | 0.620893 |

Interpretation:
- The real-data pipeline now runs end to end with a valid temporal split.
- The timestamp bug was an experiment-framework issue and materially depressed the earlier metrics.
- After the fix, the main bottleneck is still retrieval: test candidate_positive_recall is `0.098848`, so around 90% of positive labels are injected for supervision rather than naturally retrieved.
- Rankers can fit candidate-level labels, especially DIN, but they do not materially improve top-k because most positives are absent from the candidate pool.
- The remaining weakness is cold/tail recall: head positives with train popularity `31+` are retrieved at about `62%`, while cold positives are retrieved at about `0.39%`.
- The next engineering priority is recall expansion: category/author/trending candidate branches, better source weighting, longer neural retriever training, per-retriever candidate recall reporting, and batch/tensor-cache performance work.

## 14. July 2026 Hardening Update

Implemented upgrades:
- Added `metadata` retriever: tail-aware category/tag/author profile retrieval for low-popularity items with sparse collaborative evidence.
- Added `trend` retriever: time-decayed popularity with light user category/tag affinity for short-video freshness effects.
- Tuned `trend` on validation: selected `half_life_days=2.0`, `category_weight=0.4`, `tag_weight=0.2`, `popularity_floor=0.02`, `candidate_cache_k=2000`.
- Added source-weighted RRF across evaluator, recommender, candidate builder, matrix runner, and single evaluation script.
- Fixed candidate retrieval depth before train/val seen filtering so Top-K evaluation and ranker samples keep a full `candidate_pool_k` instead of shrinking after exclusions.
- Added `scripts/diagnose_candidate_recall.py` for candidate recall diagnostics by source, item popularity bucket, and user history bucket.
- Masked injected-positive `source_rank` to `0` so rankers cannot exploit out-of-pool injected positions as a shortcut.
- Updated KuaiRec config and Makefile to include metadata retrieval, weighted RRF, official raw path defaults, diagnostics, and a `health` target.

Probe results on the existing official KuaiRec subset:

| run | retrievers | NDCG@10 | Recall@10 | Recall@50 | Recall@100 | CatalogCoverage | TailRatio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tuned trend single-source | trend | 0.010996 | 0.011924 | 0.078487 | 0.149920 | 0.017907 | 0.000000 |
| tuned trend-only candidate pool | trend |  |  |  | candidate_positive_recall=0.497577 |  | positive_injection_rate=0.502423 |
| metadata single-source | metadata | 0.001349 | 0.001401 | 0.004996 | 0.016987 | 0.203795 | 0.000770 |
| unweighted fusion with metadata | fusion | 0.001634 | 0.001783 | 0.010611 | 0.017982 | 0.057196 | 0.000020 |
| weighted fusion with metadata | fusion | 0.002213 | 0.002459 | 0.012043 | 0.020229 | 0.062943 | 0.000375 |
| weighted fusion with trend | fusion | 0.004594 | 0.004951 | 0.020420 | 0.043322 | 0.055058 | 0.000132 |
| previous best single-source | sasrec | 0.004462 | 0.003482 | 0.023499 | 0.042851 | 0.060136 | 0.032148 |

Interpretation:
- Tuned Trend is now the strongest offline retriever on this KuaiRec subset, improving Recall@100 by about `3.5x` over the previous best SASRec quick run.
- Trend-only candidate generation lifts test candidate_positive_recall from the older `0.098848` to `0.497577`, cutting positive injection rate from `0.901152` to `0.502423`, with `avg_candidates=500.0` after train/val seen filtering.
- Metadata retrieval increases catalog coverage and gives the system a real side-information recall branch.
- Unweighted RRF can dilute strong sources, so source weighting is necessary once exploratory recall branches are added.
- Weighted fusion improved over older fusion rows but still does not beat Trend single-source. This is an honest result: the current default should treat Trend as a strong baseline and then tune complementary recall without diluting it.

New diagnostics:
- `reports/kuairec/candidate_recall_diagnostics.md`
- `reports/kuairec/candidate_recall_by_item_popularity.csv`
- `reports/kuairec/candidate_recall_by_source.csv`
- `reports/kuairec_metadata_probe/metrics_summary.csv`
- `reports/kuairec_trend_probe/metrics_summary.csv`
- `reports/kuairec_trend_probe/candidate_recall_diagnostics.md`
- `reports/kuairec_trend_tuned_probe/metrics_summary.csv`
- `reports/kuairec_trend_tuned_probe/candidate_recall_diagnostics.md`
- `reports/kuairec_trend_tuning_fast/trend_tuning_report.md`
- `reports/kuairec_weighted_fusion_probe/metrics_summary.csv`
- `reports/kuairec_trend_weighted_fusion_probe/metrics_summary.csv`

## 15. Resume-Safe Claims
- Current honest bullet: Built and validated a multi-stage LLM4Rec recommender on official KuaiRec subset data, including dataset adapters, multi-retriever recall, candidate-level ranker sample generation, neural rankers, offline matrix reporting, candidate recall audit, failure-case export, and LLM item-ID guardrails.
- Stronger but still safe bullet: Diagnosed real KuaiRec temporal cold-item drift through candidate recall and positive injection audits, showing that recall-stage coverage, not ranker fitting, is the current bottleneck.
- Claims that should NOT be made: Do not claim online CTR/watch-time lift, production-scale training, or DeepFM/DIN ranking improvement; the current real-data Recall/NDCG numbers are low and do not support those claims.

## 16. Ranker Calibration Hardening

Additional issue found after the Trend upgrade:
- Train ranker samples are not reliable when built directly against retrievers fitted on the same train positives: retriever artifacts filter train positives internally, so train positives can become `is_injected_positive=1` rather than live retrieved positives.
- This explains why neural rankers were not rescuing final Top-K metrics: they were partly trained on positives that would never appear in a real candidate pool.
- Added `scripts/train_gbdt_ranker.py`, a lightweight sklearn GBDT calibration ranker that excludes injected positives by default.
- Added GBDT artifact loading in the offline evaluator and `probe-kuairec-gbdt` in the Makefile.
- Added automatic `encoders.pkl` persistence when `build_ranker_candidates.py --output_dir` points to an experiment-specific feature directory.

Probe setup:
- Candidate source: tuned `trend`
- Candidate pool: `1000`
- GBDT training split: validation candidates
- GBDT evaluation split: test candidates
- Training excludes injected positives because they are audit rows, not live candidates.

Final evaluator results:

| run | candidate_pool_k | ranker | rerank_mode | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | candidate_positive_recall | positive_injection_rate |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned trend retrieval-only | 500 | none | retrieval_only | 0.011924 | 0.078487 | 0.149920 | 0.010996 | 0.497577 | 0.502423 |
| trend1000 retrieval-only | 1000 | none | retrieval_only | 0.011924 | 0.078487 | 0.149920 | 0.010996 | 0.667169 | 0.332831 |
| trend1000 + GBDT | 1000 | gbdt | ranker_only | 0.020090 | 0.086503 | 0.159580 | 0.025443 | 0.667169 | 0.332831 |

Interpretation:
- Increasing the candidate pool improves candidate availability but does not change retrieval-only Top100 because the ordering is still Trend order.
- GBDT reranking provides a real final-metric lift, especially in Top10 ranking quality, but the absolute metrics remain modest.
- The next real improvement requires event-level ranker training data or stronger user-item cross features, not simply more neural model names.

## 17. Temporal Ranker Training And Head Rerank

Further hardening:
- Added `scripts/build_temporal_ranker_train.py`.
- The script splits the original train window into prefix/suffix, fits retrievers on the prefix, and builds ranker training samples from suffix positives.
- This fixes the previous all-injected-positive train-sample failure mode: temporal train candidate positive recall is now `0.539482` instead of `0.0`.
- Added user-content cross features for GBDT: `category_overlap`, `tag_overlap`, `category_jaccard`, `tag_jaccard`, `history_category_count`, and `history_tag_count`.
- Added evaluator rerank modes such as `ranker_top_50` and `ranker_top_100`, which rerank only the head of the retrieval list and preserve the rest of the retrieval ordering.
- Updated `make probe-kuairec-gbdt` to run the temporal-train + head-rerank probe.

Final evaluator results:

| run | candidate_pool_k | ranker | rerank_mode | Recall@5 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR@10 | candidate_positive_recall | positive_injection_rate |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trend1500 retrieval-only | 1500 | none | retrieval_only | 0.004330 | 0.011924 | 0.078487 | 0.149920 | 0.010996 | 0.023785 | 0.735760 | 0.264240 |
| trend1500 + temporal GBDT | 1500 | gbdt | ranker_top_50 | 0.016049 | 0.024709 | 0.078487 | 0.149920 | 0.030136 | 0.083753 | 0.735760 | 0.264240 |
| trend1500 + temporal GBDT | 1500 | gbdt | ranker_top_100 | 0.014478 | 0.023658 | 0.093230 | 0.149920 | 0.027331 | 0.072297 | 0.735760 | 0.264240 |

Interpretation:
- `ranker_top_50` gives a real top-of-list lift: Recall@10 roughly doubles and NDCG@10 improves by about `2.7x`, while Recall@100 stays unchanged.
- Full-list GBDT reranking is still unsafe because it can damage Recall@50/100; head-only reranking is a better fit for the current ranker quality.
- Exposure-quality retrieval improves candidate coverage (`trend+exposure` candidate recall reached `0.753957`), but the current ranker could not sort those added candidates well enough to improve final Top-K. Keep it as a candidate expansion branch, not as the default serving order.
- The project is more defensible now, but the absolute KuaiRec metrics remain modest. The next major lift likely needs stronger sequence/user-item interaction modeling, better negatives from exposure logs, and possibly a proper pairwise/listwise loss.
