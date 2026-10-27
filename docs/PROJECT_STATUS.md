# Project status — 2026-10-27

Status refers to this delivered snapshot and recorded verification scope. A short real-data run establishes functionality, not convergence or production readiness.

| Component | Status | Evidence / boundary |
|---|---|---|
| Python 3.11 environment and editable installation | Fully verified | Pinned offline requirements; install and smoke logs |
| Python 3.12 clean installation | Fully verified | Independently installed declared requirements; tests and complete fixture smoke |
| Existing unit tests and added regressions | Fully verified | 40 tests; Python 3.11/3.12 logs |
| MovieLens-format fixture pipeline | Fully verified | Retrieval, EmbeddingMLP/DeepFM/DIN training, matrix and diagnostics; not official MovieLens data |
| Official KuaiRec acquisition and subset | Fully verified | Published MD5, source hashes, 1,000 users / 7,483 items / 300,000 interactions |
| Per-user historical split | Partially verified | Reproduced as historical comparison; cross-user future interactions remain possible by design |
| Global chronological split | Fully verified | Strict timestamp boundaries; separate artifacts and metrics |
| Feature encoders and temporal tree candidates | Fully verified | Prefix-only training, independent target labels, bounded-memory Parquet writer |
| Popular, Trend, ItemCF | Fully verified | Real historical/strict retrieval and candidate metrics; saved models |
| RRF / weighted RRF | Fully verified | Real 1,500-pool ablation; weights fixed in advance |
| GBDT full / Top50 / Top100 | Fully verified | Real train/val fitting and final historical/strict test metrics |
| LambdaRank full / Top50 / Top100 | Fully verified | Real train/val fitting, saved schema; current labels effectively binary |
| ExposureQuality, Metadata, TF-IDF content retrieval | Fully verified | Strict real-data metrics and save/load agreement in retriever verification |
| BPR, TwoTower, LightGCN, SASRec | Partially verified | Fixed three-epoch real-data verification; not the historical ten-epoch training protocol or convergence study |
| SentenceTransformer + FAISS semantic retrieval | Not verified / blocked | Optional packages/model weights not downloaded; TF-IDF backend verified separately |
| EmbeddingMLP / DeepFM / DIN | Smoke-tested only | Fixture fitting/inference; no real KuaiRec ranker-quality experiment |
| CTRMLP legacy/synthetic path | Smoke-tested only | Existing unit tests; not primary current pipeline |
| Exposure negative-sampling helper | Partially verified | Future-label and prefix guards tested; historical ablation not rerun |
| Business diversity/tail reranking | Smoke-tested only | Tests and fixture matrix; not primary real-data ranking table |
| Recall/NDCG/MRR/MAP and candidate audits | Fully verified | Hand calculations, independent truth and real-data invariants |
| Coverage/diversity/novelty/tail/latency | Partially verified | Real offline outputs; category-count diversity and timing scopes documented; no load benchmark |
| Failure analysis | Fully verified | Per-user Top10 misses, pool miss vs ordering miss, popularity/history buckets |
| Local real-data GBDT recommendation demo | Fully verified | Ten catalog IDs from strict model, local template explanation, no API calls |
| Template explanation and ID guard | Smoke-tested only | Deterministic tests; ID regex is not full semantic/factual validation |
| Live LLM explanation API | Not verified / blocked | No paid API invoked; not needed for recommendations |
| FastAPI service | Smoke-tested only | Local /health and /recommend on fixture models passed; no production/load verification |
| MIND adapter | Smoke-tested only | Existing adapter/group tests; no real MIND data experiment |
| KuaiRand | Not verified / blocked | No dedicated adapter or real KuaiRand experiment; historical broad support claims withdrawn |
| Docker image | Not verified / blocked | CPU dependency recipe repaired; Docker build not executed |
| Local git repository and delivery archive | Fully verified | Commit, tracked-file review, bundle and ZIP validation |
| GitHub repository upload | Not verified / blocked | Account identity available; no accessible repository/create API; no authenticated local CLI |
| GitHub Actions remote run | Not verified / blocked | Workflow configured; local equivalent passes, no remote run exists |
| Public release | Not verified / blocked | Not requested/authorized; no visibility change performed |

Main results: `reports/reproduction/REPRODUCTION_REPORT.md`. Exact commands, dataset metadata, split boundaries, training configurations and artifact hashes are tracked. Input-archive reports and portfolio materials are marked historical to avoid presenting old metrics as current evidence.

Remote delivery: no repository URL is claimed. After signing in with GitHub CLI, run `scripts/upload_private.sh`; it creates a private repository only when no origin or existing repository is found, without any force push or visibility change. A successful local test run does not imply passing GitHub Actions.
