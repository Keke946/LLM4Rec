"""Fixed three-epoch CPU verification of optional retrievers on real strict data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm4rec.retrieval import RETRIEVER_REGISTRY
from llm4rec.retrieval.base import BaseRetriever
from llm4rec.evaluation.metrics import ranking_metrics


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',default='exposure,metadata,content_faiss,bpr,two_tower,lightgcn,sasrec')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--output-dir',type=Path,default=Path('reports/reproduction/retriever_verification'))
    parser.add_argument('--model-dir',type=Path,default=Path('models/kuairec_strict/retrievers'))
    parser.add_argument("--load-only", action="store_true")
    args=parser.parse_args()
    processed=Path('data/processed/kuairec_strict')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    interactions=pd.read_parquet(processed/'interactions.parquet')
    items=pd.read_parquet(processed/'items.parquet')
    train=interactions[interactions.split=='train']
    known=interactions[(interactions.split.isin(['train','val'])) & (interactions.label==1)]
    seen={str(u):set(g.item_id.astype(str)) for u,g in known.groupby('user_id')}
    targets=interactions[(interactions.split=='test') & (interactions.label==1)]
    truth={str(u):set(g.item_id.astype(str))-seen.get(str(u),set()) for u,g in targets.groupby('user_id')}
    truth={u:r for u,r in truth.items() if r}
    rows=[]
    for name in args.models.split(','):
        kwargs={}
        if name in ['bpr','two_tower','lightgcn','sasrec']:
            kwargs.update(epochs=args.epochs,embedding_dim=32,batch_size=1024,seed=42,device='cpu')
        if name=='two_tower':kwargs['max_history_len']=50
        if name=='sasrec':kwargs['max_seq_len']=50
        path=args.model_dir/name
        start=time.perf_counter()
        if args.load_only:
            model=BaseRetriever.load(path)
            if name in ['bpr','two_tower','lightgcn','sasrec']:
                assert model.epochs==args.epochs
            train_seconds=None
        else:
            model=RETRIEVER_REGISTRY[name](**kwargs).fit(train,items)
            train_seconds=time.perf_counter()-start
            model.save(path)
        restored=BaseRetriever.load(path)
        for user in sorted(truth)[:5]:
            expected=[c.item_id for c in model.retrieve(user,top_k=100)]
            actual=[c.item_id for c in restored.retrieve(user,top_k=100)]
            assert actual==expected, f'{name}: saved/loaded predictions differ'
        metrics=[];counts=[];retrieved=0
        for user,rel in truth.items():
            candidates=restored.retrieve(user,top_k=1500+len(seen.get(user,set())))
            ids=[c.item_id for c in candidates if c.item_id not in seen.get(user,set())][:1500]
            assert len(ids)==len(set(ids))
            metrics.append(ranking_metrics(ids[:100],rel,[10,100]))
            retrieved+=len(set(ids)&rel);counts.append(len(ids))
        row=dict(model=name,epochs=kwargs.get('epochs'),training_seconds=train_seconds,
                 evaluated_users=len(truth),candidate_recall_micro=retrieved/sum(map(len,truth.values())),
                 avg_candidates=float(np.mean(counts)),backend=getattr(model,'backend','native'),
                 save_load_match_users=5,
                 **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
        rows.append(row)
        pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
        (output/'parameters.json').write_text(json.dumps(dict(seed=42,epochs=args.epochs,embedding_dim=32,
                                                            max_history_len=50,protocol='global_temporal',
                                                            model_names=args.models.split(',')),indent=2))
        print(json.dumps(row),flush=True)
        del model,restored


if __name__=='__main__':main()
