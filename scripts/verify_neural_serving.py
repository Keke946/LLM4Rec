"""Check the public recommender's head ordering against offline evaluation."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from llm4rec.recommender import LLM4RecRecommender
from llm4rec.evaluation.evaluator import OfflineEvaluator
from llm4rec.retrieval import reciprocal_rank_fusion
from scripts.evaluate_completion import load_truth


def main():
    truth,seen=load_truth();users=sorted(truth)[:3]
    evaluator=OfflineEvaluator('data/processed/kuairec_strict',retriever_dir='models/kuairec_strict/retrievers',
        features_dir='data/features/kuairec_strict_trend1500',ranker_root='models/kuairec_completion')
    recommender=LLM4RecRecommender(dataset='kuairec_strict',processed_dir='data/processed/kuairec_strict',
        retriever_dir='models/kuairec_strict/retrievers',use_llm_explanations=False,
        features_dir='data/features/kuairec_strict_trend1500',ranker_root='models/kuairec_completion')
    for name in ['embedding_mlp','deepfm','din']:
        for user in users:
            base=evaluator._load_or_fit_retriever('trend').retrieve(user,top_k=1500+len(seen.get(user,set())))
            base=evaluator._filter_candidate_objects(base,seen.get(user,set()))
            base=reciprocal_rank_fusion([base],top_k=1500,weights={'trend':1.6})
            for mode in ['ranker_top_50','ranker_top_100']:
                expected=evaluator._rerank_with_ranker(user,base,evaluator._load_ranker(name),100,mode,.7,.3,['train','val'])
                # The recommender mutates candidate scores/ranks; use fresh candidate objects.
                from dataclasses import replace
                copies=[replace(c) for c in base]
                actual=recommender._rerank_processed_candidates(user,copies,recommender._load_processed_ranker(name),
                    100,mode,.7,.3,['train','val'])
                assert [c.item_id for c in actual]==expected,(name,user,mode)
                if mode == 'ranker_top_50':
                    public=recommender.recommend(user_id=user,top_k=100,candidate_pool_k=1500,
                        retrievers=['trend'],ranker=name,rerank_mode=mode,source_weights={'trend':1.6},
                        use_llm_explainer=False,debug=True)
                    assert [row['item_id'] for row in public]==expected,(name,user,'public endpoint')
                    assert public[0]['debug']['candidate_pool_size']==1500
    path=Path('reports/reproduction/completion/neural/serving_audit.json')
    path.write_text(json.dumps(dict(models=3,users_per_model=3,scopes=['top50','top100'],
        public_recommender_offline_order_match=True,full_public_recommend_match=True,uses_paid_api=False),indent=2))
    print('Public recommender parity passed for 3 models x 3 users x 2 scopes.')


if __name__=='__main__':main()
