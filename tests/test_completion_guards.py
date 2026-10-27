import unittest
from collections import Counter
import numpy as np
import pandas as pd
from llm4rec.rerank.business import _as_list, apply_business_reranker
from llm4rec.ranking.negative_sampling import sample_exposure_aware_negatives


class CompletionGuardsTest(unittest.TestCase):
    def test_explicit_semantic_backend_cannot_be_mislabeled_tfidf(self):
        import sys
        from unittest.mock import patch
        from llm4rec.retrieval.content_faiss import ContentFaissRetriever
        train=pd.DataFrame([dict(user_id='u',item_id='i',label=1)])
        items=pd.DataFrame([dict(item_id='i',full_text='pets dogs')])
        with patch.dict(sys.modules,{'sentence_transformers':None}):
            with self.assertRaises(ImportError):
                ContentFaissRetriever(backend='semantic').fit(train,items)

    def test_parquet_category_array_preserves_individual_ids(self):
        self.assertEqual(_as_list(np.array(['a','b'])),['a','b'])
        items=['a','b','c']
        result=apply_business_reranker(items,{i:{'categories':np.array(['x'])} for i in items},Counter(),set(),mode='diversity')
        self.assertEqual(set(result),set(items))

    def test_sampling_does_not_turn_candidate_recall_into_100_percent(self):
        frame=pd.DataFrame([
            dict(user_id='u',item_id='p',label=1,rank=1,is_injected_positive=0,is_retrieved_positive=1),
            dict(user_id='u',item_id='injected',label=1,rank=2,is_injected_positive=1,is_retrieved_positive=0),
            dict(user_id='u',item_id='n',label=0,rank=3,is_injected_positive=0,is_retrieved_positive=0)])
        sampled,stats=sample_exposure_aware_negatives(frame,strategy='random_negative')
        self.assertEqual(stats['candidate_positive_recall'],.5)
        self.assertEqual(int(sampled.label.sum()),1)

    def test_missing_exposure_negative_pool_is_not_silently_replaced(self):
        frame=pd.DataFrame([dict(user_id='u',item_id='p',label=1,rank=1,is_injected_positive=0),
                            dict(user_id='u',item_id='n',label=0,rank=2,is_injected_positive=0)])
        with self.assertRaises(ValueError):
            sample_exposure_aware_negatives(frame,strategy='exposed_low_watch_negative')


if __name__=='__main__':unittest.main()
