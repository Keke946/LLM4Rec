"""Build current report and historical difference tables from recorded metrics."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/reproduction'


def markdown(frame):
    text=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        text.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(text)


def main():
    historical=pd.read_csv(REPORTS/'historical/metrics.csv')
    strict=pd.read_csv(REPORTS/'strict/metrics.csv')
    old=pd.DataFrame([
        ['retrieval_only',.012151,.011062,.023785],['gbdt_full',.012672,.015193,.040112],
        ['gbdt_top50',.026115,.029657,.079903],['gbdt_top100',.022871,.025955,.067142],
        ['lambdarank_full',.008716,.009050,.022420],['lambdarank_top50',.017210,.017448,.041612],
        ['lambdarank_top100',.014470,.014879,.036811]],columns=['experiment','Recall@10','NDCG@10','MRR@10'])
    differences=[]
    for row in old.to_dict('records'):
        actual=historical[historical.experiment==row['experiment']].iloc[0]
        for key in ['Recall@10','NDCG@10','MRR@10']:
            differences.append(dict(experiment=row['experiment'],metric=key,historical_report=row[key],
                                    repaired_historical=actual[key],absolute_difference=actual[key]-row[key],
                                    relative_difference_percent=(actual[key]/row[key]-1)*100))
    diffs=pd.DataFrame(differences);diffs.to_csv(REPORTS/'historical_differences.csv',index=False)
    columns=['experiment','Recall@10','NDCG@10','MRR@10','Recall@100','candidate_recall_micro']
    short_diffs=diffs[diffs.metric=='Recall@10']
    audit=json.loads((REPORTS/'strict/evaluation_audit.json').read_text())
    base=strict.set_index('experiment').loc['retrieval_only'];head=strict.set_index('experiment').loc['gbdt_top50']
    lifts={m:(head[m]/base[m]-1)*100 for m in ['Recall@10','NDCG@10','MRR@10']}
    report=f'''# KuaiRec repair and reproduction report

Run date: 2026-10-27. All current measurements use the official real-data archive. Historical records are preserved separately under `reports/historical/`. No synthetic or fixture metrics are used in this report.

## Outcome

The original 1,000-user / 7,483-item / 300,000-interaction subset was recovered exactly. Retrieval-only historical Recall@10, NDCG@10 and MRR@10 match the supplied report to six decimal places. Repaired temporal GBDT Top50 also recovers the historical qualitative result, with different fitted ranking scores. The stricter chronological run has lower absolute quality and still benefits from head reranking.

Under the predeclared strict protocol, GBDT Top50 improves Recall@10 by **{lifts['Recall@10']:.2f}%**, NDCG@10 by **{lifts['NDCG@10']:.2f}%**, and MRR@10 by **{lifts['MRR@10']:.2f}%** relative to the same Trend pool without reranking. These are descriptive single-seed offline comparisons, not significance or online-lift claims. LambdaRank Top50 has higher strict NDCG@10 than GBDT Top50, while GBDT Top50 has higher Recall@10. No test-driven winner selection or parameter changes followed these results.

## Data and experiment conditions

- Official KuaiRec 2.0 big matrix, downloaded from the upstream-referenced Zenodo record. Published MD5 verified; archive and extracted-input SHA-256 values are recorded in `data_source.json`.
- Keep the 1,000 most active users; uniformly sample 300,000 interactions without replacement, seed 42. Positive rate: 0.32984. Binary label: `watch_ratio >= 1.0`.
- Historical per-user split: 239,988 train / 30,006 val / 30,006 test rows. Strict global split: 240,000 / 30,000 / 30,000 rows, with no shared timestamps across boundaries.
- Historical evaluation: 987 eligible positive-test users and 9,287 unique eligible user/item positives. Strict: {audit['eligible_test_users']} users and {audit['eligible_positive_pairs']:,} eligible positives. Counts differ from positive interaction-row totals because repeated pairs and previously positive items are handled explicitly.
- The same catalog and initial interaction subset are used for both protocols. Test labels are independent of candidate tables. All eligible users remain in the metric denominator, including users with no retrieved positives.
- Main pool: Trend only, 1,500 candidates; half-life 2 days; category weight .4, tag weight .2, popularity floor .02, cache 2,000, RRF source weight 1.6. Multi-source RRF is a separate retrieval ablation, not the source of the main head-rerank result.
- Train ranker targets come from the suffix after an 80% timestamp quantile inside train. Only prefix behavior builds those candidate features. Validation candidates use train; test histories use train+val positive history capped at 50. Retrieval models remain trained on train.
- GBDT: HistGradientBoostingClassifier, 120 iterations, learning rate .07, 31 leaves, L2 .05, balanced training weights, random seed 42. sklearn may internally early-stop on a held-out part of training; external validation metrics are diagnostic and never use test.
- LambdaRank: 120 trees, learning rate .05, 31 leaves, min child samples 20, subsample .9, column sample .9, L2 .1, random state 42. Candidate labels contain no per-candidate watch ratio, so its graded option reduces to positive-vs-unlabeled relevance (2 vs 0), not a genuine graded-watch-time experiment.
- GBDT limits: 700,000 train / 500,000 validation rows. LambdaRank preserves whole user query groups within those limits. Injected positives are excluded from both train and validation model inputs. Feature schema excludes labels and audit flags.
- CPU Python 3.11.17, torch 2.6.0+cpu, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, pyarrow 19.0.1, LightGBM 4.6.0; two OpenMP/BLAS threads. Exact per-command arguments, durations and artifact hashes are in each protocol directory.

## Primary strict chronological results

{markdown(strict[columns])}

Candidate recall here is micro recall over eligible positive pairs. `metrics.csv` also contains user-macro candidate recall, precision, hit rate, MAP, coverage, diversity, novelty, tail ratio and timing scopes. Catalog/diversity metrics describe lists at the maximum reported cutoff, 100. Diversity is category IDs per recommendation and is not a bounded semantic-diversity measure.

## Historical-protocol reproduction

{markdown(historical[columns])}

Historical source values versus repaired results for Recall@10:

{markdown(short_diffs[['experiment','historical_report','repaired_historical','absolute_difference','relative_difference_percent']])}

The full three-metric comparison, with absolute and relative differences, is in `historical_differences.csv`. Original numbers are rounded to six decimals, so even an exact baseline match has a small residual relative to the rounded reference. Historical weights, exact runtime and any unrecorded tuning decisions were unavailable. This is not a claim of byte-identical model reproduction.

## Why the protocols differ

The supplied per-user protocol has cross-user chronological overlap: the latest training event occurs after the earliest test events. The strict run uses global boundaries, so this mechanism is absent. It also changes target-user composition and temporal item distribution; the lower strict score cannot be attributed solely to one leakage mechanism.

Strict Trend candidate recall is 63.73%, versus 73.58% under the historical protocol. In strict test, 3,000 of 8,952 eligible positives (33.51%) have zero positive interactions in train, and Trend retrieves none of these cold positives. The remaining warm buckets have candidate recall around 91–98%. This is a concrete cold-item bottleneck, rather than evidence that the ranker alone can solve overall recall. Under historical splitting there are 2,044 cold eligible positives instead.

At Top10, strict retrieval-only misses every positive for 879 of 928 eligible users; GBDT Top50 reduces this to 835. In both, 47 users have no positive anywhere in the 1,500 pool. `failure_cases.json` records cases and `failure_summary.csv` separates whole-pool misses from positives retrieved below the display head. Category/popularity/history diagnostics provide additional context; the reason labels are heuristics rather than causal explanations.

Equal RRF across Popular/Trend/ItemCF is weaker than Trend alone in both runs. The frozen weighted fusion improves over equal RRF but still underperforms Trend alone. Extra sources do not automatically improve a strong retrieval path. Full GBDT reranking harms strict Recall@10 and Recall@100; Top50 helps the head while preserving the candidate order after position 50. Head Recall@100 equality is an asserted invariant, not an independently learned gain.

## Methodology audit

1. Strict train max timestamp < val min; val max < test min. No validation/test interactions fit the strict retrievers or temporal training features. Static catalog/category metadata is transductive and assumed known; upload-time availability remains unmodeled.
2. Test is rejected as either training or validation input by both tree-training CLIs. No tuning search was performed. All seven primary ranking variants were fixed before final test evaluation.
3. Independent processed test truth supplies metric denominators. Candidates are unique, belong to the catalog and contain no seen positive train/val items. Repeated/seen positives are excluded consistently from eligible truth. Cold or zero-hit users remain included.
4. Injected positives are audit rows only: they never enter tree fitting, validation prediction, or test recommendation lists. Pool-size features are computed before injection.
5. Ten users' cached Top1500 lists match independently loaded live Trend retrieval. Model-facing tree features share the same vectorized implementation as online rows; history length is capped consistently at 50. The local demo uses the strict fitted GBDT model.
6. Negative-sampling code rejects future-label filtering and uses prefix-only exposure information. The historical negative-sampling ablation is not rerun or endorsed here.
7. Recall/NDCG/MRR/MAP have hand-calculated regression cases; duplicate recommendation IDs and invalid cutoffs are rejected. Recall@100 invariance holds for Top50/Top100 variants.

## Evidence and limits

See `evaluation_audit.json`, `candidate_meta.json`, split statistics, diagnostics, model/validation configurations, command logs and manifests in each protocol directory. The strict run is the supported primary offline estimate. Historical improvements remain informative only within their protocol.

Additional retriever verification uses fixed three-epoch neural training, independently reported in `retriever_verification/metrics.csv`; it is a functionality/quality snapshot, not a converged comparison with tuned baselines. Semantic FAISS/SentenceTransformer, real-data neural ranker experiments, live paid LLM APIs, full-data scaling, online serving performance and remote CI are outside the verified scope unless the project status says otherwise.

GitHub upload is blocked in this session: the connected account identifies Keke946 but exposes no accessible repositories or repository-creation operation, and no authenticated local GitHub CLI is available. Local git delivery is complete separately; no GitHub URL or remote CI success is asserted. Public visibility has not been enabled.
'''
    (REPORTS/'REPRODUCTION_REPORT.md').write_text(report)
    for protocol,dataset in [('historical','kuairec'),('strict','kuairec_strict')]:
        destination=REPORTS/protocol/'model_evidence';destination.mkdir(exist_ok=True)
        for model in ['gbdt','lambdarank']:
            source=ROOT/f'models/{dataset}_trend1500'/model
            for name in ['config.json','metrics.json','feature_schema.json']:
                path=source/name
                if path.exists():(destination/f'{model}_{name}').write_text(path.read_text())
    print('Wrote report, differences and model evidence.')


if __name__=='__main__':main()
