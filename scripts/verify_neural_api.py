"""Local HTTP smoke for real-data neural recommendations; no LLM API is invoked."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm4rec.recommender import LLM4RecRecommender
from scripts.evaluate_completion import load_truth


def main():
    os.chdir(ROOT)
    truth,_=load_truth();user=sorted(truth)[0]
    payload=dict(user_id=user,top_k=10,candidate_pool_k=1500,retrievers=['trend'],ranker='din',
                 rerank_mode='ranker_top_50',source_weights={'trend':1.6},debug=True)
    recommender=LLM4RecRecommender(dataset='kuairec_strict',processed_dir='data/processed/kuairec_strict',
        retriever_dir='models/kuairec_strict/retrievers',features_dir='data/features/kuairec_strict_trend1500',
        ranker_root='models/kuairec_completion',use_llm_explanations=False)
    expected=recommender.recommend(**payload,use_llm_explainer=False)
    env={**os.environ,'LLM4REC_DATASET':'kuairec_strict','LLM4REC_PROCESSED_DIR':'data/processed/kuairec_strict',
         'LLM4REC_RETRIEVER_DIR':'models/kuairec_strict/retrievers',
         'LLM4REC_FEATURES_DIR':'data/features/kuairec_strict_trend1500',
         'LLM4REC_RANKER_ROOT':'models/kuairec_completion','LLM4REC_HOST':'127.0.0.1','LLM4REC_PORT':'8766'}
    process=subprocess.Popen([sys.executable,'scripts/serve_api.py'],env=env,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                health=json.load(urllib.request.urlopen('http://127.0.0.1:8766/health',timeout=2));break
            except OSError:time.sleep(.25)
        else:raise RuntimeError('Local service failed to start')
        request=urllib.request.Request('http://127.0.0.1:8766/recommend',data=json.dumps(payload).encode(),
                                      headers={'Content-Type':'application/json'})
        result=json.load(urllib.request.urlopen(request,timeout=30))
        assert [r['item_id'] for r in result['results']]==[r['item_id'] for r in expected]
        assert len(result['results'])==10 and result['results'][0]['debug']['candidate_pool_size']==1500
        out=Path('reports/reproduction/completion/neural/api_smoke.json')
        out.write_text(json.dumps(dict(health=health,request=payload,response=result,
                                       offline_order_match=True,paid_api_used=False),indent=2))
        print('Real-data neural HTTP smoke passed.')
    finally:
        process.terminate();process.wait(timeout=10)


if __name__=='__main__':main()
