# Additional module acceptance — 2026-10-27

Scope: finish the existing strict KuaiRec module checks after the accepted initial repair. Live LLM APIs, GitHub upload and remote CI were deferred explicitly by the owner. Previous baseline measurements are preserved. Docker execution remains an environment block; all other stages listed below produced real-data results.

## Fixed protocol

Same 1,000 users, 7,483 videos, 300,000 interactions, strict global time split, watch-ratio threshold 1.0, 928 eligible test users and 8,952 eligible positive pairs. The candidate ranker experiments all use the same 1,500-item Trend pool. Ground truth comes from the independent processed test table. No new algorithm family, model-selection search or test-based tuning was performed.

`configs/kuairec_completion.yaml`, `frozen_spec.yaml`, `execution.json`, model configurations and logs record choices. Neural test scoring followed the fitting of all three neural rankers. Ten-epoch retriever choices and negative strategies were specified before their test evaluation. Additional models do not replace the predeclared GBDT baseline simply because one test metric is higher.

## Neural ranking: real training and test

Dense features use a fixed log1p transform, saved in model config and applied inside each model at training and serving. All injected rows are removed before reproducible row sampling. Up to 700,000 train and 500,000 validation rows, batches 1,024, 32-dimensional embeddings, most recent 50 positive history items, learning rate .001, maximum 10 epochs, patience 2 on validation AUC, gradient norm clipping 5.0, seed 42. Early stopping is the declared complete fitting procedure; it does not mean every model ran ten epochs. Tests were not used for checkpoint selection.

| model | completed_epochs | best_epoch | validation_auc | train_rows | val_rows |
| --- | --- | --- | --- | --- | --- |
| embedding_mlp | 3 | 1 | 0.574615 | 700000 | 500000 |
| deepfm | 6 | 4 | 0.528255 | 700000 | 500000 |
| din | 3 | 1 | 0.583661 | 700000 | 500000 |

All predeclared ranking scopes:

| experiment | Recall@10 | NDCG@10 | MRR@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| retrieval_only | 0.006745 | 0.005966 | 0.010812 | 0.094395 |
| embedding_mlp_full | 0.014333 | 0.019013 | 0.048995 | 0.077135 |
| embedding_mlp_top50 | 0.014029 | 0.018058 | 0.050673 | 0.094395 |
| embedding_mlp_top100 | 0.014576 | 0.019117 | 0.052168 | 0.094395 |
| deepfm_full | 0.005748 | 0.006467 | 0.015597 | 0.055281 |
| deepfm_top50 | 0.011514 | 0.011986 | 0.029372 | 0.094395 |
| deepfm_top100 | 0.010550 | 0.011217 | 0.028045 | 0.094395 |
| din_full | 0.014961 | 0.020649 | 0.058397 | 0.079708 |
| din_top50 | 0.011739 | 0.016961 | 0.050255 | 0.094395 |
| din_top100 | 0.013841 | 0.019328 | 0.055175 | 0.094395 |

The retained GBDT Top50 reference is Recall@10 0.010793, NDCG@10 0.010182, MRR@10 0.021441. Neural Top50/Top100 preserve Recall@100; full reranking can change deep recall. Results are single-seed descriptive measurements. These compact models are not claimed to be tuned, converged optima or significant improvements over the full literature.

Cached test-feature predictions match the independent offline feature builder for three users per model. The public recommender also matches offline Top50/Top100 ordering for three users per model. Candidate membership, recent-history truncation, capped history-length features and requested head scope agree. The local neural demo and real-data API smoke use fitted weights with no LLM calls.

## Complete ten-epoch neural retrieval configuration

| model | epochs | Recall@10 | NDCG@10 | MRR@10 | Recall@100 | candidate_recall_micro | save_load_match_users |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bpr | 10 | 0.002653 | 0.002818 | 0.007154 | 0.014974 | 0.219839 | 5 |
| two_tower | 10 | 0.000698 | 0.000823 | 0.002185 | 0.013504 | 0.179625 | 5 |
| lightgcn | 10 | 0.001397 | 0.001523 | 0.003515 | 0.020006 | 0.216488 | 5 |
| sasrec | 10 | 0.007168 | 0.007974 | 0.019326 | 0.066232 | 0.513852 | 5 |

All four models completed ten epochs on strict train, with seed 42, embedding dimension 32, batch 1,024 and CPU. TwoTower/SASRec use history length 50; LightGCN has two propagation layers. Learning rate .01 follows the pre-existing training CLI. The earlier three-epoch SASRec check used its class default .001, so comparing the old and new SASRec rows is not a controlled epoch-only ablation. More epochs are not assumed to improve performance. Saved/loaded Top100 lists agree for five users per model. No retriever weights were selected using test results.

## Repaired negative-sampling ablation

Each strategy fits a separate fixed 120-tree LambdaRank model against the same validation and final test pools. Only training-prefix exposures are used for negative typing; future val/test labels are rejected. Random negatives are sampled uniformly within the live retrieval-negative pool, not the entire catalog. Exposure/hard/popularity labels follow the existing taxonomy. Specific empty pools fail explicitly. The exposure-only pool is much smaller than the requested random-negative budget; therefore sample count and negative type are confounded, and these rows must not be read as a pure equal-budget type ablation. Mixed sampling reports actual type counts and fills shortages from remaining eligible live negatives.

| strategy | positives | negatives | candidate_recall_before_sampling | future_labels_used | empty_pool_fallback |
| --- | --- | --- | --- | --- | --- |
| random_negative | 8459 | 169180 | 0.567718 | False | False |
| exposed_low_watch_negative | 8459 | 12347 | 0.567718 | False | False |
| retrieved_hard_negative | 8459 | 169180 | 0.567718 | False | False |
| mixed_negative | 8459 | 169180 | 0.567718 | False | False |

| experiment | Recall@10 | NDCG@10 | MRR@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| retrieval_only | 0.006745 | 0.005966 | 0.010812 | 0.094395 |
| random_negative_full | 0.006544 | 0.008549 | 0.023300 | 0.075200 |
| random_negative_top50 | 0.010891 | 0.011509 | 0.026544 | 0.094395 |
| random_negative_top100 | 0.008842 | 0.009584 | 0.023789 | 0.094395 |
| exposed_low_watch_negative_full | 0.004745 | 0.005646 | 0.014278 | 0.055568 |
| exposed_low_watch_negative_top50 | 0.010795 | 0.011209 | 0.025916 | 0.094395 |
| exposed_low_watch_negative_top100 | 0.010460 | 0.010866 | 0.024218 | 0.094395 |
| retrieved_hard_negative_full | 0.006967 | 0.005681 | 0.011006 | 0.030789 |
| retrieved_hard_negative_top50 | 0.005894 | 0.004768 | 0.008902 | 0.094395 |
| retrieved_hard_negative_top100 | 0.005642 | 0.004288 | 0.007821 | 0.094395 |
| mixed_negative_full | 0.005606 | 0.006411 | 0.016101 | 0.061264 |
| mixed_negative_top50 | 0.009989 | 0.009610 | 0.019574 | 0.094395 |
| mixed_negative_top100 | 0.010949 | 0.010967 | 0.024631 | 0.094395 |

Candidate recall remains measured against all original target positives before injection exclusion and sampling. It is not redefined as 100% merely because only retrieved positives enter fitting. Exposed low-watch history can conflict with later target positives; target positives keep their positive labels and are never selected as negatives. Unexposed target-window nonpositives are still unlabeled candidates, not known dislikes.

## Semantic content retrieval

Official caption metadata covers 7483 evaluated catalog videos. A fixed revision of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encodes official cover text, captions, topic tags and category names into normalized 384-dimensional vectors. This is a local pretrained encoder, not a paid API or a newly fine-tuned LLM.

| model | backend | Recall@10 | NDCG@10 | MRR@10 | Recall@100 | candidate_recall_micro |
| --- | --- | --- | --- | --- | --- | --- |
| multilingual_caption_faiss | faiss | 0.001261 | 0.001544 | 0.003881 | 0.010877 | 0.153262 |

The run requires the semantic backend and verifies `faiss`, preventing a TF-IDF fallback from being mislabeled semantic. Exact FAISS inner products agree numerically with normalized-vector dot products for five users. Saved/loaded Top100 recommendations agree for five users. Source revision, caption hash, coverage, backend and artifact hashes are retained. The official CSV has four unquoted caption line breaks: the strict parser joins only recognized 3-field/8-field continuations, logs those repairs, and verifies all 10,728 unique video IDs without skipping records. The serialized FAISS backend is rebuilt from stored vectors on load; scores retain actual cosine/inner-product values rather than artificial rank scores.

Captions are assumed-static transductive side information. Their publication-time availability in the historical logging period is not established. They are used in a separate retrieval view, leaving the baseline data/features unchanged.

## Business reranking on real data

| experiment | Recall@10 | NDCG@10 | MRR@10 | Recall@100 | CatalogCoverage | Diversity | TailRatio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt_top50_business_none | 0.010793 | 0.010182 | 0.021441 | 0.094395 | 0.006281 | 0.573707 | 0.000000 |
| gbdt_top50_business_diversity | 0.012131 | 0.011570 | 0.025357 | 0.094395 | 0.006147 | 0.591164 | 0.000000 |
| gbdt_top50_business_diversity_tail | 0.012131 | 0.011570 | 0.025357 | 0.094395 | 0.006147 | 0.591164 | 0.000000 |

All modes receive the same GBDT Top50 Top100 input head. They preserve its 100-item membership. Catalog coverage, diversity and tail ratio here describe Top10 display lists. Diversity means distinct encrypted category IDs per recommendation, not semantic distance, and may exceed one for multi-tag videos. The category cap is a soft penalty. Author information is absent from this adapter path, so author diversity is not meaningfully evaluated. No business weights were tuned on test.

## Additional correctness repairs

- Neural training now supports injected-row exclusion, tensor caching, bounded batch inference, fixed dense-feature transforms and validation histories.
- Serving uses the most recent history window, its capped length and the requested head permutation. It expands source retrieval before seen filtering, preventing pool shrinkage relative to evaluation. Missing requested ranker artifacts fail explicitly instead of silently returning retrieval-only results.
- Model/features locations can be specified for the API. Head scores describe ranker outputs; they are not an unused fusion score.
- Business reranking correctly handles Arrow/NumPy list-valued category metadata. Its service head is bounded consistently with offline evaluation.
- Negative typing is vectorized, excludes injected rows from popularity thresholds, rejects empty-pool substitution, and retains the original candidate-recall denominator.
- Semantic fitting can require its backend, preserves real similarity scores and restores the FAISS index after loading.
- Docker context excludes raw data, weights, credentials, caches and unrelated evidence dumps.

## Remaining boundaries

Docker status: **blocked**. Docker executable/engine unavailable; rootless user namespaces are denied.

This host cannot execute a real container build. The recipe and commands are provided for a supported Docker host; a syntax/configuration check is not represented as a successful build.

Live LLM API integration, GitHub upload and remote CI were intentionally not performed in this run. MIND/KuaiRand experiments, production load tests, online A/B evaluation, multi-seed uncertainty and new recommendation architectures remain outside this KuaiRec acceptance scope. The completion report supersedes older smoke-only/partial labels for the modules actually validated here; it does not erase these explicit limits.
