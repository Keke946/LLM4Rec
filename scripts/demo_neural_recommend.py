"""Run an accepted neural ranker on local strict KuaiRec data without LLM APIs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm4rec.recommender import LLM4RecRecommender
from scripts.evaluate_completion import load_truth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranker',choices=['embedding_mlp','deepfm','din'],default='din')
    parser.add_argument('--user-id')
    args=parser.parse_args()
    truth,_=load_truth();user=args.user_id or sorted(truth)[0]
    recommender=LLM4RecRecommender(dataset='kuairec_strict',processed_dir='data/processed/kuairec_strict',
        retriever_dir='models/kuairec_strict/retrievers',features_dir='data/features/kuairec_strict_trend1500',
        ranker_root='models/kuairec_completion',use_llm_explanations=False)
    result=recommender.recommend(user_id=user,top_k=10,candidate_pool_k=1500,retrievers=['trend'],
        ranker=args.ranker,rerank_mode='ranker_top_50',source_weights={'trend':1.6},use_llm_explainer=False,debug=True)
    print(json.dumps(dict(user_id=user,ranker=args.ranker,results=result),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
