"""Evaluate frozen candidate pools against independent raw split ground truth."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from llm4rec.evaluation.metrics import (ranking_metrics, catalog_coverage, category_diversity,
                                       tail_ratio, novelty)
from llm4rec.retrieval import RETRIEVER_REGISTRY, reciprocal_rank_fusion
from llm4rec.retrieval.base import BaseRetriever
from scripts.run_ltr_ablation import _ranker_scores, _ordered_items_for_group

CUTS = [5, 10, 20, 50, 100]


def positives(frame, splits):
    data = frame[(frame.split.isin(splits)) & (frame.label == 1)]
    return {str(user): set(group.item_id.astype(str)) for user, group in data.groupby('user_id')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['processed-dir', 'features-dir', 'models', 'retrievers', 'output-dir']:
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.processed_dir/'meta.json').read_text())
    if meta.get('is_sample_fallback') or meta.get('is_simulated') or meta['num_interactions'] < 300000:
        raise ValueError('Real-data evaluation requires the verified 300K KuaiRec subset.')
    interactions = pd.read_parquet(args.processed_dir/'interactions.parquet')
    items = pd.read_parquet(args.processed_dir/'items.parquet')
    train = interactions[interactions.split == 'train']
    seen = positives(interactions, ['train','val'])
    raw_truth = positives(interactions, ['test'])
    truth = {user: rel - seen.get(user, set()) for user, rel in raw_truth.items()}
    truth = {user: rel for user, rel in truth.items() if rel}
    popularity = Counter(train[train.label == 1].item_id.astype(str))
    threshold = np.quantile(list(popularity.values()), 0.2)
    tails = {item for item, count in popularity.items() if count <= threshold}
    catalog = set(items.item_id.astype(str))
    item_meta = items.set_index('item_id').to_dict('index')
    total_relevant = sum(map(len, truth.values()))
    rows, failures = [], []

    def score(name, recs, pools, latencies=None, **extra):
        measures = []
        for user, rel in truth.items():
            rec = recs.get(user, [])
            pool = pools.get(user, [])
            assert len(pool) == len(set(pool)), 'Duplicate candidate IDs'
            assert not set(pool) & seen.get(user, set()), 'Seen positive returned'
            assert set(pool) <= catalog, 'Unknown catalog ID returned'
            assert set(rec) <= set(pool), 'Reranking introduced an out-of-pool item'
            measures.append(ranking_metrics(rec, rel, CUTS))
            if not set(rec[:10]) & rel:
                present = rel & set(pool)
                reason = 'not_retrieved' if not present else 'retrieved_but_low_rank'
                failures.append(dict(experiment=name, user_id=user, reason=reason,
                                     relevant_count=len(rel), retrieved_positives=len(present),
                                     top10=rec[:10], missed_positive_examples=sorted(rel)[:5]))
        hits = sum(len(set(pools.get(user, [])) & rel) for user, rel in truth.items())
        row = dict(experiment=name, **{key: float(np.mean([m[key] for m in measures])) for key in measures[0]},
                   candidate_recall_micro=hits/total_relevant,
                   candidate_recall_macro=float(np.mean([len(set(pools.get(u, [])) & r)/len(r) for u,r in truth.items()])),
                   evaluated_users=len(truth), relevant_items=total_relevant,
                   avg_candidates=float(np.mean([len(pools.get(u, [])) for u in truth])),
                   CatalogCoverage=catalog_coverage(recs, catalog), Diversity=category_diversity(recs, item_meta),
                   TailRatio=tail_ratio(recs, tails), Novelty=novelty(recs, popularity), **extra)
        if latencies:
            row.update(latency_p50_ms=float(np.median(latencies)), latency_p95_ms=float(np.quantile(latencies, .95)))
        rows.append(row)

    # Baselines share the same labels, seen policy, pool size, and catalog.
    retrievers = {}
    for name in ['popular', 'trend', 'itemcf']:
        artifact = args.retrievers / name
        model = BaseRetriever.load(artifact) if artifact.exists() else RETRIEVER_REGISTRY[name]().fit(train, items)
        if not artifact.exists():
            model.save(artifact)
        retrievers[name] = model
    strategies = [('popular', ['popular'], {}), ('itemcf', ['itemcf'], {}),
                  ('rrf_popular_trend_itemcf', ['popular','trend','itemcf'], {}),
                  ('weighted_rrf_popular_trend_itemcf', ['popular','trend','itemcf'],
                   {'popular': .4,'trend': 1.6,'itemcf': .8})]
    for name, sources, weights in strategies:
        pools, recs, latencies = {}, {}, []
        for user in truth:
            start = time.perf_counter()
            excluded = seen.get(user, set())
            lists = []
            for source in sources:
                candidates = retrievers[source].retrieve(user, top_k=1500+len(excluded))
                filtered = [c for c in candidates if c.item_id not in excluded]
                for rank, candidate in enumerate(filtered, 1):
                    candidate.rank = rank
                lists.append(filtered)
            merged = reciprocal_rank_fusion(lists, top_k=1500, weights=weights)
            pools[user] = [c.item_id for c in merged]
            recs[user] = pools[user][:100]
            latencies.append((time.perf_counter()-start)*1000)
        score(name, recs, pools, latencies, latency_scope='warm retrieval and fusion, no disk I/O')
        print(name, rows[-1]['Recall@10'], flush=True)
    # Project only features required by tree rankers; long sequence tensors stay on disk.
    path = args.features_dir/'test_rank_samples.parquet'
    excluded = {'history_item_indices','retrieval_source_indices','timestamp','split'}
    columns = [c for c in pq.read_schema(path).names if c not in excluded]
    full = pd.read_parquet(path, columns=columns)
    live = full[full.is_injected_positive == 0].copy().reset_index(drop=True)
    del full
    assert not live.duplicated(['user_id','item_id']).any()
    for user, group in live.groupby('user_id'):
        rel = truth.get(str(user), set())
        assert set(group[group.label == 1].item_id.astype(str)) == set(group.item_id.astype(str)) & rel
        assert (group.candidate_pool_size == len(group)).all(), 'Label-dependent pool size'
    pools = {str(u): g.sort_values('rank').item_id.astype(str).tolist() for u,g in live.groupby('user_id')}
    # Validate that cached samples agree with deployed retrieval, independently of labels.
    for user in sorted(truth)[:10]:
        excluded = seen.get(user, set())
        candidates = retrievers['trend'].retrieve(user, top_k=1500+len(excluded))
        expected = [c.item_id for c in candidates if c.item_id not in excluded][:1500]
        assert expected == pools.get(user, []), 'Cached and live retrieval differ'
    score('retrieval_only', {u: p[:100] for u,p in pools.items()}, pools)
    prediction_cache = {}
    for model_name in ['gbdt','lambdarank']:
        start = time.perf_counter()
        predictions = _ranker_scores(live, args.models, model_name, prediction_cache)
        if predictions is None:
            raise FileNotFoundError(f'Missing fitted {model_name}')
        predict_seconds = time.perf_counter()-start
        live['_ranker_score'] = predictions
        for scope, mode in [('full','ranker_only'),('top50','ranker_top_50'),('top100','ranker_top_100')]:
            recs = {str(u): _ordered_items_for_group(group, mode, .7, .3)[:100] for u,group in live.groupby('user_id')}
            score(f'{model_name}_{scope}', recs, pools, batch_feature_and_prediction_seconds=predict_seconds)
        prediction_cache.clear()
    frame = pd.DataFrame(rows)
    reference = frame[frame.experiment == 'retrieval_only'].iloc[0]
    for name in ['gbdt_top50','gbdt_top100','lambdarank_top50','lambdarank_top100']:
        assert np.isclose(frame[frame.experiment == name]['Recall@100'].iloc[0], reference['Recall@100'])
    frame.to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(failures).to_json(output/'failure_cases.json', orient='records', indent=2)
    failure_counts = pd.DataFrame(failures).groupby(['experiment','reason']).size().unstack(fill_value=0)
    failure_counts.to_csv(output/'failure_summary.csv')
    audit = dict(data_meta=meta, raw_positive_test_users=len(raw_truth), eligible_test_users=len(truth),
                 raw_positive_pairs=sum(map(len,raw_truth.values())), eligible_positive_pairs=total_relevant,
                 training_time_max=int(train.timestamp.max()),
                 validation_time_min=int(interactions[interactions.split=='val'].timestamp.min()),
                 validation_time_max=int(interactions[interactions.split=='val'].timestamp.max()),
                 test_time_min=int(interactions[interactions.split=='test'].timestamp.min()),
                 cached_live_top1500_match_users=min(10,len(truth)),
                 all_candidate_lists_unique=True, all_candidates_in_catalog=True,
                 no_seen_positive_candidates=True, injected_rows_excluded_from_predictions=True,
                 independent_ground_truth=True, head_rerank_preserves_recall100=True,
                 missing_or_cold_test_users_included=True,
                 negative_label_meaning='Not a positive in the target window; unexposed items may be false negatives.')
    if meta['split_strategy'] == 'global_temporal':
        assert audit['training_time_max'] < audit['validation_time_min']
        assert audit['validation_time_max'] < audit['test_time_min']
    (output/'evaluation_audit.json').write_text(json.dumps(audit, indent=2))
    print(frame[['experiment','Recall@10','NDCG@10','MRR@10','Recall@100','candidate_recall_micro']].to_string(index=False))


if __name__ == '__main__':
    main()
