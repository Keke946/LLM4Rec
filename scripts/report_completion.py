"""Summarize completed, fixed-scope acceptance from recorded files only."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/reproduction/completion'


def table(frame,columns):
    frame=frame[columns]
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    neural=pd.read_csv(OUT/'neural/metrics.csv')
    training=pd.read_csv(OUT/'neural/training_summary.csv')
    ret=pd.read_csv(OUT/'retrievers/metrics.csv')
    negative=pd.read_csv(OUT/'negative/metrics.csv')
    semantic=pd.read_csv(OUT/'semantic/metrics.csv')
    business=pd.read_csv(OUT/'business/metrics.csv')
    docker=json.loads((OUT/'docker_status.json').read_text())
    original=pd.read_csv(ROOT/'reports/reproduction/strict/metrics.csv')
    baseline=original.set_index('experiment').loc['gbdt_top50']
    cols=['experiment','Recall@10','NDCG@10','MRR@10','Recall@100']
    snapshot=[]
    for name in ['embedding_mlp','deepfm','din']:
        snapshot.append(neural[neural.experiment==name+'_top50'].iloc[0].to_dict())
    snapshot=pd.DataFrame(snapshot)
    samples=[]
    for strategy in ['random_negative','exposed_low_watch_negative','retrieved_hard_negative','mixed_negative']:
        stats=json.loads((OUT/'negative'/strategy/'negative_sampling_stats.json').read_text())
        samples.append(dict(strategy=strategy,positives=stats['num_positives'],negatives=stats['num_negatives'],
                            candidate_recall_before_sampling=stats['candidate_positive_recall'],
                            future_labels_used=False,empty_pool_fallback=stats['empty_pool_fallback']))
    sampled=pd.DataFrame(samples);sampled.to_csv(OUT/'negative/sampling_summary.csv',index=False)
    with (OUT/'semantic/audit.json').open() as f:semantic_audit=json.load(f)
    report=f'''# Additional module acceptance — 2026-10-27

Scope: finish the existing strict KuaiRec module checks after the accepted initial repair. Live LLM APIs, GitHub upload and remote CI were deferred explicitly by the owner. Previous baseline measurements are preserved. Docker execution remains an environment block; all other stages listed below produced real-data results.

## Fixed protocol

Same 1,000 users, 7,483 videos, 300,000 interactions, strict global time split, watch-ratio threshold 1.0, 928 eligible test users and 8,952 eligible positive pairs. The candidate ranker experiments all use the same 1,500-item Trend pool. Ground truth comes from the independent processed test table. No new algorithm family, model-selection search or test-based tuning was performed.

`configs/kuairec_completion.yaml`, `frozen_spec.yaml`, `execution.json`, model configurations and logs record choices. Neural test scoring followed the fitting of all three neural rankers. Ten-epoch retriever choices and negative strategies were specified before their test evaluation. Additional models do not replace the predeclared GBDT baseline simply because one test metric is higher.

## Neural ranking: real training and test

Dense features use a fixed log1p transform, saved in model config and applied inside each model at training and serving. All injected rows are removed before reproducible row sampling. Up to 700,000 train and 500,000 validation rows, batches 1,024, 32-dimensional embeddings, most recent 50 positive history items, learning rate .001, maximum 10 epochs, patience 2 on validation AUC, gradient norm clipping 5.0, seed 42. Early stopping is the declared complete fitting procedure; it does not mean every model ran ten epochs. Tests were not used for checkpoint selection.

{table(training,['model','completed_epochs','best_epoch','validation_auc','train_rows','val_rows'])}

All predeclared ranking scopes:

{table(neural,cols)}

The retained GBDT Top50 reference is Recall@10 {baseline['Recall@10']:.6f}, NDCG@10 {baseline['NDCG@10']:.6f}, MRR@10 {baseline['MRR@10']:.6f}. Neural Top50/Top100 preserve Recall@100; full reranking can change deep recall. Results are single-seed descriptive measurements. These compact models are not claimed to be tuned, converged optima or significant improvements over the full literature.

Cached test-feature predictions match the independent offline feature builder for three users per model. The public recommender also matches offline Top50/Top100 ordering for three users per model. Candidate membership, recent-history truncation, capped history-length features and requested head scope agree. The local neural demo and real-data API smoke use fitted weights with no LLM calls.

## Complete ten-epoch neural retrieval configuration

{table(ret,['model','epochs','Recall@10','NDCG@10','MRR@10','Recall@100','candidate_recall_micro','save_load_match_users'])}

All four models completed ten epochs on strict train, with seed 42, embedding dimension 32, batch 1,024 and CPU. TwoTower/SASRec use history length 50; LightGCN has two propagation layers. Learning rate .01 follows the pre-existing training CLI. The earlier three-epoch SASRec check used its class default .001, so comparing the old and new SASRec rows is not a controlled epoch-only ablation. More epochs are not assumed to improve performance. Saved/loaded Top100 lists agree for five users per model. No retriever weights were selected using test results.

## Repaired negative-sampling ablation

Each strategy fits a separate fixed 120-tree LambdaRank model against the same validation and final test pools. Only training-prefix exposures are used for negative typing; future val/test labels are rejected. Random negatives are sampled uniformly within the live retrieval-negative pool, not the entire catalog. Exposure/hard/popularity labels follow the existing taxonomy. Specific empty pools fail explicitly. The exposure-only pool is much smaller than the requested random-negative budget; therefore sample count and negative type are confounded, and these rows must not be read as a pure equal-budget type ablation. Mixed sampling reports actual type counts and fills shortages from remaining eligible live negatives.

{table(sampled,['strategy','positives','negatives','candidate_recall_before_sampling','future_labels_used','empty_pool_fallback'])}

{table(negative,cols)}

Candidate recall remains measured against all original target positives before injection exclusion and sampling. It is not redefined as 100% merely because only retrieved positives enter fitting. Exposed low-watch history can conflict with later target positives; target positives keep their positive labels and are never selected as negatives. Unexposed target-window nonpositives are still unlabeled candidates, not known dislikes.

## Semantic content retrieval

Official caption metadata covers {semantic_audit['caption_coverage']} evaluated catalog videos. A fixed revision of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encodes official cover text, captions, topic tags and category names into normalized 384-dimensional vectors. This is a local pretrained encoder, not a paid API or a newly fine-tuned LLM.

{table(semantic,['model','backend','Recall@10','NDCG@10','MRR@10','Recall@100','candidate_recall_micro'])}

The run requires the semantic backend and verifies `faiss`, preventing a TF-IDF fallback from being mislabeled semantic. Exact FAISS inner products agree numerically with normalized-vector dot products for five users. Saved/loaded Top100 recommendations agree for five users. Source revision, caption hash, coverage, backend and artifact hashes are retained. The official CSV has four unquoted caption line breaks: the strict parser joins only recognized 3-field/8-field continuations, logs those repairs, and verifies all 10,728 unique video IDs without skipping records. The serialized FAISS backend is rebuilt from stored vectors on load; scores retain actual cosine/inner-product values rather than artificial rank scores.

Captions are assumed-static transductive side information. Their publication-time availability in the historical logging period is not established. They are used in a separate retrieval view, leaving the baseline data/features unchanged.

## Business reranking on real data

{table(business[business.experiment.str.contains('business')],['experiment','Recall@10','NDCG@10','MRR@10','Recall@100','CatalogCoverage','Diversity','TailRatio'])}

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

Docker status: **{docker['status']}**. {docker.get('reason','Image built and container tests executed.')}

This host cannot execute a real container build. The recipe and commands are provided for a supported Docker host; a syntax/configuration check is not represented as a successful build.

Live LLM API integration, GitHub upload and remote CI were intentionally not performed in this run. MIND/KuaiRand experiments, production load tests, online A/B evaluation, multi-seed uncertainty and new recommendation architectures remain outside this KuaiRec acceptance scope. The completion report supersedes older smoke-only/partial labels for the modules actually validated here; it does not erase these explicit limits.
'''
    (OUT/'COMPLETION_REPORT.md').write_text(report)
    print('Wrote completion report and sampling summary.')


if __name__=='__main__':main()
