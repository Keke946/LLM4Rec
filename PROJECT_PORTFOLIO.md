# LLM4Rec project portfolio

An offline short-video recommender built around the official KuaiRec big matrix. The repaired project covers data provenance, chronological splitting, retrieval/fusion, temporal candidate-level ranking, evaluation and reproducible delivery.

- Real subset: 1,000 users, 7,483 videos, 300,000 interactions; watch-completion threshold >= 1.0.
- Main ranking: Trend Top1500 with GBDT and LambdaRank, full/Top50/Top100 scopes. RRF and weighted RRF are separate retrieval ablations.
- Strict global chronological GBDT Top50: Recall@10 0.010793, NDCG@10 0.010182, MRR@10 0.021441, Recall@100 0.094395. Relative Recall/NDCG gains vs the same retrieval pool: 60.0% / 70.7%.
- Historical-protocol retrieval metrics match the supplied reference at six decimal places. Historical and strict metrics are reported separately.
- 50 tests and the fixture pipeline pass in Python 3.11 and a separate clean Python 3.12 environment. Neural rankers, ten-epoch neural retrievers, prefix-safe negative ablations, caption/FAISS and business reranking have real-data checks.
- LLMs do not select item IDs. Live LLM API usage is outside the verified scope.

See [the measured report](reports/reproduction/REPRODUCTION_REPORT.md), [module status](docs/PROJECT_STATUS.md), [reproduction commands](docs/REPRODUCE.md) and [repair evidence](docs/REPAIR_AND_REPRODUCTION_LOG.md). Current results are single-seed offline observations, with an exposure-biased active-user subset and a substantial cold-item bottleneck. No online CTR or production-readiness claim is made. GitHub upload and remote CI are deferred by the owner; actual Docker execution is blocked by the host.
