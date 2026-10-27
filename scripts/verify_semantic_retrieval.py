"""Verify caption-backed multilingual SentenceTransformer + FAISS, with no LLM API."""
from pathlib import Path
import json
import hashlib
import sys
import zipfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from llm4rec.retrieval.content_faiss import ContentFaissRetriever
from llm4rec.retrieval.base import BaseRetriever
from llm4rec.evaluation.metrics import ranking_metrics
from scripts.evaluate_completion import load_truth


def main():
    out=Path('reports/reproduction/completion/semantic');out.mkdir(parents=True,exist_ok=True)
    archive=Path('data/raw/kuairec_official/KuaiRec.zip')
    raw=Path('data/raw/kuairec_official/extracted/kuairec_caption_category.csv')
    if not raw.exists():
        with zipfile.ZipFile(archive) as bundle:
            matches=[name for name in bundle.namelist() if name.endswith('/kuairec_caption_category.csv')]
            if len(matches)!=1:raise ValueError('Ambiguous/missing official caption CSV')
            raw.write_bytes(bundle.read(matches[0]))
    captions=pd.read_csv(raw).fillna('');captions['item_id']=captions.video_id.astype(str)
    items=pd.read_parquet('data/processed/kuairec_strict/items.parquet')
    columns=['manual_cover_text','caption','topic_tag','first_level_category_name','second_level_category_name']
    columns=[c for c in columns if c in captions]
    text=captions.set_index('item_id')[columns].astype(str).agg('. '.join,axis=1).to_dict()
    items['full_text']=items.item_id.map(text).fillna('')
    coverage=int(items.item_id.isin(text).sum())
    assert coverage==len(items),'Caption metadata must cover the evaluated catalog'
    interactions=pd.read_parquet('data/processed/kuairec_strict/interactions.parquet')
    train=interactions[interactions.split=='train']
    start=time.perf_counter()
    model=ContentFaissRetriever(model_name='data/raw/semantic_encoder',backend='semantic').fit(train,items)
    assert model.backend=='faiss',model.backend
    assert model.embeddings.shape==(len(items),384)
    np.testing.assert_allclose(np.linalg.norm(model.embeddings,axis=1),1,atol=1e-5)
    build_seconds=time.perf_counter()-start
    path=Path('models/kuairec_completion/semantic');model.save(path)
    restored=BaseRetriever.load(path);assert restored.backend=='faiss'
    truth,seen=load_truth()
    for user in sorted(truth)[:5]:
        assert [c.item_id for c in model.retrieve(user,top_k=100)]==[c.item_id for c in restored.retrieve(user,top_k=100)]
        query=model._query_embedding(user,None,model._profile_text(user))
        distances,indices=model.index.search(query,len(items))
        exact=model.embeddings@query.reshape(-1)
        np.testing.assert_allclose(distances[0],exact[indices[0]],rtol=1e-4,atol=1e-5)
    metrics=[];hits=0;latencies=[]
    for user,rel in truth.items():
        start=time.perf_counter()
        ids=[c.item_id for c in restored.retrieve(user,top_k=1500+len(seen.get(user,set())))
             if c.item_id not in seen.get(user,set())][:1500]
        latencies.append((time.perf_counter()-start)*1000)
        assert len(ids)==len(set(ids)) and not set(ids)&seen.get(user,set())
        hits+=len(set(ids)&rel);metrics.append(ranking_metrics(ids[:100],rel,[5,10,20,50,100]))
    row=dict(model='multilingual_caption_faiss',backend=model.backend,evaluated_users=len(truth),
             caption_coverage=coverage,build_seconds=build_seconds,
             candidate_recall_micro=hits/sum(map(len,truth.values())),latency_p50_ms=float(np.median(latencies)),
             **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
    pd.DataFrame([row]).to_csv(out/'metrics.csv',index=False)
    checks=dict(caption_file_sha256=hashlib.file_digest(raw.open('rb'),'sha256').hexdigest(),
                caption_fields=columns,caption_coverage=coverage,normalized_embeddings=True,
                exact_inner_product_agreement_users=5,saved_loaded_top100_match_users=5,
                backend='faiss',model_source=json.loads((out/'model_source.json').read_text()),
                catalog_assumption='Transductive static captions; publication-time availability not modeled.')
    (out/'audit.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':main()
