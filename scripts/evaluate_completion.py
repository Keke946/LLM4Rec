"""Final evaluation of predeclared completion experiments against strict test truth."""
from __future__ import annotations
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from llm4rec.evaluation.metrics import ranking_metrics, category_diversity, catalog_coverage, tail_ratio
from llm4rec.features.tensor_samples import candidate_tensors
from llm4rec.ranking import RANKER_REGISTRY
from llm4rec.ranking.base import RankerConfig
from scripts.run_ltr_ablation import _ranker_scores, _ordered_items_for_group

FEATURES=Path('data/features/kuairec_strict_trend1500')


def load_truth():
    interactions=pd.read_parquet('data/processed/kuairec_strict/interactions.parquet')
    seen={str(u):set(g.item_id.astype(str)) for u,g in interactions[(interactions.split.isin(['train','val'])) & (interactions.label==1)].groupby('user_id')}
    truth={str(u):set(g.item_id.astype(str))-seen.get(str(u),set()) for u,g in interactions[(interactions.split=='test') & (interactions.label==1)].groupby('user_id')}
    return {u:r for u,r in truth.items() if r},seen


def summarize(recs,truth,experiment):
    results=[ranking_metrics(recs.get(u,[]),r,[5,10,20,50,100]) for u,r in truth.items()]
    return dict(experiment=experiment,evaluated_users=len(truth),
                **{k:float(np.mean([r[k] for r in results])) for k in results[0]})


def load_neural(path):
    config=json.loads((path/'config.json').read_text())
    allowed={f.name for f in fields(RankerConfig)}
    model=RANKER_REGISTRY[config['model']](RankerConfig(**{k:v for k,v in config.items() if k in allowed}))
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True));model.eval()
    return model,config


def neural_scores(frame,model,batch_size=2048):
    # Tensorize bounded blocks, rather than holding a second full tensor dataset.
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(frame),batch_size):
            batch=candidate_tensors(frame.iloc[start:start+batch_size],model.config.max_history_len)
            scores.extend(model(batch).cpu().numpy().tolist())
    return pd.Series(scores,index=frame.index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['neural','negative','business'],required=True)
    args=parser.parse_args()
    out=Path('reports/reproduction/completion')/args.stage;out.mkdir(parents=True,exist_ok=True)
    truth,seen=load_truth()
    exclude={'timestamp','split'}
    if args.stage!='neural':exclude|={'history_item_indices','retrieval_source_indices'}
    path=FEATURES/'test_rank_samples.parquet'
    frame=pd.read_parquet(path,columns=[c for c in pq.read_schema(path).names if c not in exclude])
    frame=frame[frame.is_injected_positive==0].reset_index(drop=True)
    pools={str(u):g.sort_values('rank').item_id.astype(str).tolist() for u,g in frame.groupby('user_id')}
    for u,pool in pools.items():
        assert len(pool)==1500 and len(pool)==len(set(pool)) and not set(pool)&seen.get(u,set())
    rows=[summarize({u:p[:100] for u,p in pools.items()},truth,'retrieval_only')]
    audit={'stage':args.stage,'independent_truth':True,'injected_excluded':True,'test_users':len(truth),
           'predeclared_spec':'configs/kuairec_completion.yaml','head_recall100_preserved':True}
    if args.stage=='neural':
        names=['embedding_mlp','deepfm','din']
    elif args.stage=='negative':
        names=['random_negative','exposed_low_watch_negative','retrieved_hard_negative','mixed_negative']
    else:names=['gbdt']
    for name in names:
        start=time.perf_counter()
        if args.stage=='neural':
            model,config=load_neural(Path('models/kuairec_completion')/name)
            scores=neural_scores(frame,model)
            # Compare cached tensor prediction with the serving feature builder.
            from llm4rec.evaluation.evaluator import OfflineEvaluator
            evaluator=OfflineEvaluator('data/processed/kuairec_strict',
                retriever_dir='models/kuairec_strict/retrievers',features_dir=FEATURES,
                ranker_root='models/kuairec_completion')
            for user in sorted(truth)[:3]:
                candidates=evaluator._load_or_fit_retriever('trend').retrieve(user,top_k=1500+len(seen.get(user,set())))
                candidates=evaluator._filter_candidate_objects(candidates,seen.get(user,set()))
                from llm4rec.retrieval import reciprocal_rank_fusion
                candidates=reciprocal_rank_fusion([candidates],top_k=1500,weights={'trend':1.6})
                actual=evaluator._rerank_with_ranker(user,candidates,evaluator._load_ranker(name),100,'ranker_top_50',.7,.3,['train','val'])
                group=frame[frame.user_id==user].copy();group['_ranker_score']=scores.loc[group.index]
                expected=_ordered_items_for_group(group,'ranker_top_50',.7,.3)[:100]
                assert actual==expected, f'{name}: serving and cached predictions differ for {user}'
        elif args.stage=='negative':
            scores=_ranker_scores(frame,Path('models/kuairec_completion/negatives')/name,'lambdarank',{})
        else:
            scores=_ranker_scores(frame,Path('models/kuairec_strict_trend1500'),'gbdt',{})
        assert scores is not None and np.isfinite(scores).all()
        elapsed=time.perf_counter()-start
        frame['_ranker_score']=scores
        scopes=[('full','ranker_only'),('top50','ranker_top_50'),('top100','ranker_top_100')]
        for scope,mode in scopes:
            recs={str(u):_ordered_items_for_group(g,mode,.7,.3)[:100] for u,g in frame.groupby('user_id')}
            row=summarize(recs,truth,f'{name}_{scope}')
            row['batch_prediction_seconds']=elapsed
            if scope!='full':assert np.isclose(row['Recall@100'],rows[0]['Recall@100'])
            rows.append(row)
            if args.stage=='business' and scope=='top50':
                from collections import Counter
                from llm4rec.rerank import apply_business_reranker
                interactions=pd.read_parquet('data/processed/kuairec_strict/interactions.parquet')
                popularity=Counter(interactions[(interactions.split=='train') & (interactions.label==1)].item_id.astype(str))
                items=pd.read_parquet('data/processed/kuairec_strict/items.parquet');meta=items.set_index('item_id').to_dict('index')
                threshold=np.quantile(list(popularity.values()),.2);tail={i for i,c in popularity.items() if c<=threshold}
                for business in ['none','diversity','diversity_tail']:
                    adjusted={u:apply_business_reranker(ids,meta,popularity,tail,mode=business,top_k=100) for u,ids in recs.items()}
                    for u,ids in adjusted.items():assert set(ids)==set(recs[u])
                    m=summarize(adjusted,truth,f'gbdt_top50_business_{business}')
                    m.update(CatalogCoverage=catalog_coverage({u:r[:10] for u,r in adjusted.items()},set(meta)),
                             Diversity=category_diversity({u:r[:10] for u,r in adjusted.items()},meta),
                             TailRatio=tail_ratio({u:r[:10] for u,r in adjusted.items()},tail))
                    rows.append(m)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        print(name,rows[-1],flush=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':main()
