# Repair and reproduction log — 2026-10-27

Input: `LLM4Rec(2).zip`, the attachment actually supplied in this task. Windows backslash separators and trailing backslash directory entries were normalized during safe extraction. Source was repaired in place; existing model families and module boundaries were retained.

The initial host used Python 3.12 without torch/pyarrow/lightgbm. Compilation passed but unittest discovery had dependency import errors. A separate CPython 3.11.17 environment was installed and the original 31 tests passed. Logs are in `reports/reproduction/logs`; initial missing-dependency errors are preserved.

| Original problem | Repair and reason | Verification |
|---|---|---|
| GBDT config and matrix defaults train on val, validate on test | Require `train/val` in tree-training entry points; repair config and defaults. Build prefix/suffix temporal training candidates | Training-split rejection regression; real command logs |
| Candidate pool size included label-dependent injected positives | Freeze pool size before injection; both tree rankers see only live rows | Candidate-builder regression and real-data assertions |
| Exposure negative sampling excluded future val/test positives | Reject future-label selection; pass only temporal prefix interactions | Regression verifies rejection and future-label independence; no new negative-sampling benchmark claimed |
| Per-user split permits cross-user future events | Retain historical comparison; add global temporal split as primary strict protocol | Timestamp-boundary regression and strict evaluation audit |
| Old cached-pool evaluator inferred truth from sample rows | Read truth independently from processed test interactions | New evaluation verifies labels and counts all eligible test users, including zero-hit users |
| Missing ranker could silently evaluate retrieval-only | Raise when requested artifacts are absent | Entry-point inspection and real tree-model load |
| MAP@10 divided by returned-list length for short lists | Use min(number of relevant items, 10), independent of short returned lists | Short-list hand-calculated regression; real pools all exceed 100 so recorded real metrics unchanged |
| Dense negative sampling could return a positive after 20 retries | Exact eligible fallback; explicit error when no negative exists | Dense-catalog and saturated-user regression |
| GBDT online history length used full count versus capped training history | Match the 50-item candidate feature policy | Feature-row parity regression; recorded serving scope is 50 |
| Content retriever lost history when restored | Reconstruct text lookup from persisted item IDs and text | Save/load profile and recommendation regression |
| MovieLens diagnostics crashed on no source rows | Return a stable empty source table | Empty-pool regression and complete smoke target |
| Tiny MovieLens fixture had no positive test labels | Add authored positive test examples; label fixture clearly | Nonempty test candidates and successful evaluation; fixture metrics never reported as real data |
| Explicit big/small matrix request could fall back to the other matrix | Respect explicit matrix requests and fail if missing | Big-request/small-only regression |
| Missing raw data silently fell back to tiny fixtures | Require explicit `--allow_sample_fallback` | Strict reproduction uses `--no_sample_fallback`, real-data size and provenance guards |
| Candidate generator accumulated millions of Python rows | Stream per-user batches to Parquet, preserve schema | Multi-batch regression and 1.5M-scale candidate generation |
| GBDT feature conversion materialized huge record dictionaries | Share vectorized candidate feature implementation; project needed Parquet columns | Online/vectorized feature parity regression and CPU training |
| Report directory ignored wholesale | Track curated current evidence and separately labeled historical records | Git tracked-file review |
| Unbounded dependencies and missing build backend | Pin offline dependencies, add installable setuptools package, separate optional integrations | Editable install log, Python 3.11 tests and smoke |
| Latency measured retrieval only despite reranking | Include ranking/business ordering in the general evaluator; label standalone batch prediction timing separately | Code inspection; current report explicitly states timing scope |
| ItemCF accumulation depended on set iteration order | Iterate sorted histories | Fixed seed/hash environment and saved artifact checks |

Fixed experiment choices: seed 42; 1,000 most active users; 300,000 random interactions; watch ratio >= 1.0; 1,500 Trend candidates; two-day decay; 80% prefix inside train for ranker targets; GBDT 120 iterations / learning rate .07 / 31 leaves; LambdaRank 120 trees / learning rate .05 / 31 leaves. No test-driven tuning or attempts to match historical ranking improvements were performed. See numbered logs and manifests for all actual arguments.

The original 1,000 / 7,483 / 300,000 data scale was recovered exactly. Historical metrics remain reference observations, not acceptance thresholds. Historical and repaired metrics are not interchangeable because the original procedure had methodology risks and original weights/environment are absent.

Remaining boundaries: no online evaluation, uncertainty intervals, full-catalog availability timestamps, production deployment, GPU validation, or paid LLM API test. The big matrix is exposure-biased and negatives are not known dislikes. The per-user protocol is retained only as a historical comparison. Neural models and optional integrations have the verification status recorded in `PROJECT_STATUS.md`. Remote publication status is separate from local verification.

The MAP short-list correction does not change recorded real-data results: every main and auxiliary evaluated pool has 1,500 items and ranking lists contain 100 items. Python 3.11/3.12 regression tests were rerun after this fix. Both FastAPI health and recommendation endpoints also passed a local fixture smoke check.
