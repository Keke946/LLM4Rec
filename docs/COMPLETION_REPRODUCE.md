# Reproduce the additional module acceptance

Start from the primary strict reproduction in `REPRODUCE.md`. Its processed tables and 1,500-candidate temporal features must exist. Large generated files and model weights stay local.

The fixed plan is in `configs/kuairec_completion.yaml`; `docs/COMPLETION_PLAN.md` explains the evaluation boundaries. No live LLM API or GitHub operation is part of this runner.

```bash
# Core environment, after completing the primary strict reproduction.
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42

# Install semantic packages separately to keep the core dependency environment unchanged.
python3.11 -m venv .venv-semantic
.venv-semantic/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-semantic/bin/python -m pip install -r requirements.txt -r requirements-semantic.txt

# Download the pinned public encoder before offline verification.
.venv-semantic/bin/python scripts/download_semantic_encoder.py

python scripts/run_completion.py --semantic-python .venv-semantic/bin/python
```

Stage commands can be run separately:

```bash
python scripts/run_completion.py --stage neural
python scripts/run_completion.py --stage retrievers
python scripts/run_completion.py --stage negative
python scripts/run_completion.py --stage semantic --semantic-python .venv-semantic/bin/python
python scripts/run_completion.py --stage business
python scripts/run_completion.py --stage docker
```

`--resume` reuses completed checkpoints in the completion model directories. Only use it for artifacts created from the same frozen specification and primary dataset. The runner checks neural injection/normalization/max-epoch settings and retriever epoch count before reuse; it is not a general model registry or complete arbitrary-artifact compatibility check.

Additional reports live under `reports/reproduction/completion/`. `execution.json` records executed subprocess commands and timings. Numbered logs are retained. Neural `training_history.json` and model configurations record actual epochs and checkpoint selection. The initial three neural fits were invoked directly with the same declared arguments; their full stdout and saved configurations are included, and the resumed runner reuses them rather than pretending it retrained them.

The negative-sampling inputs remain in the same live candidate pool. Changing the sampling distribution is not guaranteed to improve ranking. The semantic experiment uses official captions as assumed-static transductive metadata, separately from the base encrypted-category view. No future watch labels are provided to the encoder.

For Docker, install/start a compatible engine and run:

```bash
docker build -t llm4rec:acceptance .
docker run --rm -e OMP_NUM_THREADS=2 llm4rec:acceptance python -m unittest discover -s tests
docker run --rm -e OMP_NUM_THREADS=2 llm4rec:acceptance make smoke-movielens
```

This host has no usable Docker engine and rejects rootless namespaces. The runner records that environment block instead of declaring the image built. Docker execution must still be performed on a supported host before marking it passed.

Local neural API demo after acceptance:

```bash
python scripts/demo_neural_recommend.py --ranker din
LLM4REC_DATASET=kuairec_strict \
LLM4REC_FEATURES_DIR=data/features/kuairec_strict_trend1500 \
LLM4REC_RANKER_ROOT=models/kuairec_completion \
python scripts/serve_api.py
```

For the `/recommend` request, use retrievers `["trend"]`, candidate_pool_k 1500, a neural ranker name, rerank_mode `ranker_top_50` and source_weights `{"trend": 1.6}`. The service selects local IDs and uses template explanations; no LLM API is needed. `scripts/verify_neural_serving.py` compares public/offline ordering.
