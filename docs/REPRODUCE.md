# Reproduce from a clean environment

Use Linux with Python 3.11, approximately 10 GB RAM and 5 GB free disk. CPU execution is sufficient. The recorded run uses two OpenMP/BLAS threads. Runtime depends on CPU and the public dataset download; consult per-command timings in the run manifests rather than assuming GPU training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 HF_HUB_OFFLINE=1
python -m unittest discover -s tests
make smoke-movielens
python scripts/download_kuairec.py
python scripts/reproduce_kuairec.py --protocol strict
python scripts/reproduce_kuairec.py --protocol historical
```

`requirements-lock.txt` captures every package in the tested environment. The main experiment environment also contained the unused optional `openai` client; a second Python 3.12 environment installed only declared requirements and passed all tests and the complete smoke target. The pinned primary requirements are sufficient for offline experiments; semantic encoders and API explanations are optional. No API key or model download is required. Training uses the CPU PyTorch wheel. On other platforms, install the appropriate official PyTorch wheel and record the difference.

The two reproduction commands use separate processed/features/model directories, generate 1,500 Trend candidates, and build temporal training samples from a prefix/suffix split inside train. They fit GBDT and LambdaRank on train with validation from val. Both models and all rerank scopes are fixed before the test scoring command. Training or validation with `test` is rejected. This is a single-seed engineering reproduction, not a hyperparameter search or statistical significance study.

Outputs live under `reports/reproduction/{strict,historical}`:

- `metrics.csv`: popular, ItemCF, RRF, weighted RRF, Trend baseline, GBDT/LambdaRank full/top50/top100.
- `evaluation_audit.json`: independent truth, filtering, temporal boundaries, cached/live retrieval agreement and head-rerank invariants.
- `failure_cases.json`, `failure_summary.csv` and candidate bucket/source diagnostics.
- Numbered command logs, `commands.json` and `run_manifest.json`: exact arguments, environment, timings and artifact hashes.

Generated raw data, features and weights stay in ignored directories. SHA-256 hashes in the manifest let a new run check these local artifacts without publishing them. `--evaluate-only` regenerates evaluation reports from existing local artifacts; it must not be used to tune against the test set. Rerunning a protocol recreates its derived outputs and retrains Trend.

The historical protocol uses the original per-user 80/10/10 split. It can include other users' training interactions later than a target user's evaluation period and therefore is not a strict chronological estimate. The strict protocol uses global timestamp boundaries at 80% and 90%, keeping ties on the earlier side. Its training, validation and test times do not overlap. The same initial subset is used in both protocols.

Metrics are macro-averaged over users with eligible test positives. Recall divides by all eligible test positives for that user, including items absent from the candidate pool. Seen positive items from train+val are excluded; low-watch exposures are not filtered. Candidate recall is reported both as pooled-positive micro recall and user macro recall. Ranking negatives are unlabeled candidates in a target window, not guaranteed dislikes. Injected positives are audit-only for the tree experiments. Head reranking only permutes the first 50/100 candidates; its Recall@100 must equal retrieval-only.

Optional real-data retrieval checks: `make verify-retrievers` fits ExposureQuality, Metadata, TF-IDF Content, BPR, TwoTower, LightGCN and SASRec on strict train, with fixed three-epoch neural settings. It reports test metrics once and checks saved/loaded recommendations for five users per model. This does not retrain the main rankers or select fusion weights.
