> Historical material from the supplied archive; superseded by README.md, docs/PROJECT_STATUS.md and reports/reproduction/REPRODUCTION_REPORT.md. Historical metrics and improvements are not current verified claims.

# LLM4Rec Experiment Notes

## Positioning

LLM4Rec is a real-data hybrid recommender project. The main video path is KuaiRec/KuaiRand, MIND is the NLP/news recommendation path, MovieLens small is a smoke test, and synthetic data is only a fallback demo/unit-test path.

The local retrieval/ranking pipeline is the source of truth. LLMs are allowed to rewrite queries, summarize profiles, call tools, and explain already selected items, but they never generate recommendation item IDs.

## Canonical Data Schema

Items:

```text
item_id, title, description, categories, tags, author_id, duration, full_text
```

Interactions:

```text
user_id, item_id, timestamp, label, watch_ratio, impression_id, source, split
```

Dataset adapters:

- MovieLens: ratings become labels with `positive_rating_threshold`, default 4.0.
- KuaiRec: watch ratio becomes labels with `positive_watch_ratio_threshold`, default 1.0.
- MIND: clicked impression items are positives; non-clicked impression items are negatives.

## Split and History Policy

The default split is per-user temporal:

- train: earliest interactions
- val: middle interactions
- test: latest interactions

MIND uses group-aware temporal split with `user_id + impression_id` as the group key, so rows from the same impression cannot be split across train/val/test.

Ranker history policy:

- train samples use previous train positives only.
- val samples use train positives plus previous val positives.
- test samples use train plus val positives by default.
- online/evaluator reranking follows `--filter_seen_splits`; default `train,val` gives train+val history at test time, while `train` is kept for ablation.

Seen-item filtering:

- default: filter positives from `train,val`
- ablation: `--filter_seen_splits train`

## Candidate Pool and Top-K Logic

The evaluator separates candidate generation from final display:

- `--candidate_pool_k`: merged candidate pool before ranker reranking, default 500
- `--cutoffs`: metric cutoffs, default 5,10,20,50,100
- `--final_top_k`: final product-style display count, default 10

Recall@20/50/100 are computed from recommendation lists long enough for those cutoffs. They are not computed from truncated top10 output.

## Retrieval and Ranking

Retrievers:

- Popular: global positive-item popularity baseline.
- Trend: time-decayed global popularity with light category/tag affinity. It captures short-video freshness and should be treated as a strong baseline, not a decorative feature.
- ItemCF: item-item co-occurrence from training positives.
- ContentFaiss: content/text retrieval with SentenceTransformer + FAISS, with TF-IDF fallback.
- Metadata: tail-aware category/tag/author profile retrieval. It is designed to give cold and low-popularity items a side-information path when collaborative signals are sparse.
- BPR-MF: pairwise matrix factorization with DataLoader batches, sampled negatives, GPU support, and validation Recall@50/100/200 logging.
- TwoTower: user embedding plus pooled history embedding, item embedding tower, sampled negatives, exported item embeddings, optional FAISS ANN index, and validation Recall@50/100/200 logging.
- LightGCN: normalized user-item graph propagation with BPR loss and validation Recall@50/100/200 logging.
- SASRec: sequential recommendation baseline with user behavior sequence, positional embedding, Transformer self-attention blocks and next-item sampled-negative training.

Rankers:

- EmbeddingMLP: sparse embedding MLP baseline.
- DeepFM: first-order, FM second-order, and deep towers.
- DIN: candidate-conditioned attention over user history.

Ranker features:

- sparse: user, item, author, category, tag, duration bucket, retrieval source
- sequence: positive item history
- dense: retrieval_score, source_rank, source_count, item_popularity, user_activity_level, history_len

Injected positives are audit/supervision rows, not retrieval-source rows. They keep `is_injected_positive=1` for reporting, but the model-facing `source_rank` is masked to `0` and no `label_positive` source token is emitted.

## Reranking Modes

The project keeps three explicit modes:

- `retrieval_only`: normalized retrieval score only
- `ranker_only`: sigmoid ranker logit only
- `retrieval_ranker_fusion`: `alpha * normalized_retrieval_score + beta * sigmoid(ranker_logit)`

Default mode is `retrieval_ranker_fusion` with `retrieval_weight=0.7` and `ranker_weight=0.3`. This is intentional: if DeepFM/DIN are undertrained, the ranker should not blindly overwrite strong retrieval ordering. Retrieval-only and ranker-only rows remain visible in reports.

Candidate fusion uses source-weighted RRF. Default weight `1.0` preserves standard RRF behavior; KuaiRec can optionally upweight stronger sources such as SASRec/LightGCN and downweight exploratory sources such as Metadata/Popular. Source weights are recorded in `metrics_summary.csv`, `experiment_matrix.md`, and `experiment_report.md`.

## Business Reranking

The optional post-rank layer supports:

- `none`
- `diversity`
- `diversity_tail`

It applies category diversity, category cap, same-author penalty, mild tail boost and novelty-aware ordering. Reports include NDCG@10, Coverage, Diversity and TailRatio before/after business rerank.

## LLM Role and Safety

LLM is measurable through explanation safety:

- hallucinated item ID rate
- unsupported claim rate
- template fallback rate

LLM never generates item IDs. It only explains locally selected items and falls back to a template when an explanation mentions an item ID outside the allowed candidate list.

## Artifact Policy

Retriever artifacts are directory-based and version-safe:

```text
config.json
item_ids.json
train_seen_by_user.json
state.pkl / state.pt / state.npz
```

Popular and ItemCF save minimal dictionaries. BPR/TwoTower/LightGCN save Torch `state_dict` plus ID mappings and histories. Evaluator/recommender catch retriever load failures, log a warning, and refit from processed train data.

Generated raw/processed/model/report artifacts are ignored by git. Keep only lightweight docs, configs, scripts and tests in the repository.

## KuaiRec Main Pipeline

Full data:

```bash
python scripts/prepare_dataset.py --dataset kuairec --raw_dir data/raw/kuairec --output_dir data/processed/kuairec --positive_watch_ratio_threshold 1.0
python scripts/build_features.py --dataset kuairec --processed_dir data/processed/kuairec --output_dir data/features/kuairec --max_history_len 50
python scripts/train_retriever.py --dataset kuairec --model popular
python scripts/train_retriever.py --dataset kuairec --model itemcf
python scripts/train_retriever.py --dataset kuairec --model bpr --epochs 10 --batch_size 1024
python scripts/train_retriever.py --dataset kuairec --model two_tower --epochs 10 --batch_size 1024
python scripts/train_retriever.py --dataset kuairec --model lightgcn --epochs 10 --batch_size 1024
python scripts/train_retriever.py --dataset kuairec --model sasrec --epochs 10 --batch_size 1024 --max_history_len 50
python scripts/train_retriever.py --dataset kuairec --model metadata
python scripts/train_ranker.py --dataset kuairec --model deepfm --epochs 10 --batch_size 1024
python scripts/train_ranker.py --dataset kuairec --model din --epochs 10 --batch_size 1024
python scripts/run_experiment_matrix.py --dataset kuairec --config configs/kuairec.yaml
```

Subset:

```bash
python scripts/prepare_dataset.py --dataset kuairec --raw_dir data/raw/kuairec --output_dir data/processed/kuairec --positive_watch_ratio_threshold 1.0 --sample_users 500 --sample_items 5000 --max_interactions 200000
```

`meta.json` records whether the run is full data or subset.

## Experiment Matrix

```bash
python scripts/run_experiment_matrix.py --dataset kuairec
```

Outputs:

```text
reports/kuairec/metrics_summary.csv
reports/kuairec/experiment_matrix.md
reports/kuairec/failure_cases.json
reports/kuairec/experiment_report.md
reports/kuairec/candidate_recall_diagnostics.md
reports/kuairec/candidate_recall_by_item_popularity.csv
reports/kuairec/candidate_recall_by_source.csv
```

The matrix report includes dataset statistics, label definition, split strategy, retriever comparison, ranker comparison, feature/fusion ablation, cold-start notes, tail analysis, latency p50/p95 and known limitations.

For recall-stage debugging, run:

```bash
python scripts/diagnose_candidate_recall.py --dataset kuairec --split test
```

This reports candidate positive recall, positive injection rate, source-level retrieved-positive credit, and item-popularity/history buckets. It is the first place to look when Recall@K is weak, because rankers cannot recover positives absent from the candidate pool.

Latest KuaiRec subset finding: adding and tuning the Trend retriever raised retrieval-only `Recall@100` to `0.149920`, compared with the previous best SASRec row at `0.042851`. The selected Trend parameters are `half_life_days=2.0`, `category_weight=0.4`, `tag_weight=0.2`, `popularity_floor=0.02`, and `candidate_cache_k=2000`. Trend-only candidate generation also raised test candidate positive recall to `0.497577`, reducing positive injection rate to `0.502423`. Candidate retrieval now expands before filtering train/val seen items, so the evaluated pool remains at `avg_candidates=500.0`. This shows the previous bottleneck was missing a strong freshness-aware recall path.

The next hardening pass found that the ranker training split is also a bottleneck: retrievers fitted on train filter train positives internally, so train ranker samples can become all injected positives. This makes neural rankers learn from non-live positive rows. The current GBDT calibration probe therefore trains on validation candidates and evaluates on test candidates, excluding injected positives from training. With `candidate_pool_k=1000`, trend-only candidate positive recall reaches `0.667169`, injection rate drops to `0.332831`, and GBDT ranker-only reranking improves final test metrics from `Recall@10=0.011924 / NDCG@10=0.010996 / Recall@100=0.149920` to `Recall@10=0.020090 / NDCG@10=0.025443 / Recall@100=0.159580`. The metrics are still modest, but the lift is from the final evaluator rather than an oracle candidate-pool metric.

The latest hardening pass replaces that calibration-only setup with a more defensible temporal ranker-training path. `scripts/build_temporal_ranker_train.py` now splits the original train window into prefix/suffix, fits retrievers on prefix, and builds ranker training labels from suffix positives. This fixes the all-injected train-positive issue: temporal train candidate positive recall is `0.539482` instead of `0.0`. Candidate-side GBDT features now include user-content cross features (`category_overlap`, `tag_overlap`, `category_jaccard`, `tag_jaccard`, history category/tag counts). Because the temporal GBDT is still weak as a full-list sorter, evaluator supports `ranker_top_N` modes that rerank only the head of the retrieval list and preserve the rest of the retrieval ordering. On the official KuaiRec subset, `trend + temporal GBDT + ranker_top_50` improves final test metrics from `Recall@10=0.011924 / NDCG@10=0.010996 / MRR@10=0.023785` to `Recall@10=0.024709 / NDCG@10=0.030136 / MRR@10=0.083753` while keeping `Recall@100=0.149920` unchanged. This is a real top-of-list lift, but the absolute quality is still not strong enough to claim production-grade accuracy.

## Current Limitations

- KuaiRec requires local raw data; the repository cannot ship large raw files.
- MovieLens is a smoke test, not a video recommendation benchmark.
- Neural models are compact local implementations, not distributed production training jobs.
- The project still lacks feature store, model registry, online A/B testing, monitoring, and production recall service deployment.
- Cold-start and tail analysis now include explicit candidate diagnostics, but the metadata/tail branch still needs dataset-specific source-weight tuning and online validation.
