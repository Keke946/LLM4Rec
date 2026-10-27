"""Show local real-data recommendations from the strict Trend + GBDT model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm4rec.evaluation.evaluator import OfflineEvaluator
from llm4rec.retrieval import reciprocal_rank_fusion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id',default=None)
    args=parser.parse_args()
    evaluator=OfflineEvaluator('data/processed/kuairec_strict',
        retriever_dir='models/kuairec_strict/retrievers',
        features_dir='data/features/kuairec_strict_trend1500',
        ranker_root='models/kuairec_strict_trend1500')
    positives=evaluator.test[evaluator.test.label==1]
    user=args.user_id or sorted(positives.user_id.astype(str).unique())[0]
    seen=evaluator._positive_items_by_user(['train','val']).get(user,set())
    candidates=evaluator._load_or_fit_retriever('trend').retrieve(user,top_k=1500+len(seen))
    candidates=evaluator._filter_candidate_objects(candidates,seen)
    candidates=reciprocal_rank_fusion([candidates],top_k=1500,weights={'trend':1.6})
    bundle=evaluator._load_ranker('gbdt')
    if bundle is None:raise FileNotFoundError('Run the strict reproduction before this demo.')
    selected=evaluator._rerank_with_ranker(user,candidates,bundle,10,'ranker_top_50',.7,.3,['train','val'])
    assert not set(selected)&seen
    result=dict(user_id=user,model='Trend + GBDT Top50',items=[dict(
        rank=rank,item_id=item,title=evaluator.item_meta[item]['title'],
        category_ids=list(evaluator.item_meta[item].get('categories',[])),
        explanation='Selected from local Trend candidates using the fitted GBDT head reranker.')
        for rank,item in enumerate(selected,1)])
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
