import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import torch

from llm4rec.recommender import LLM4RecRecommender
from llm4rec.retrieval.base import Candidate


class ServingHeadRerankTest(unittest.TestCase):
    def test_missing_requested_ranker_cannot_silently_fall_back(self):
        recommender=object.__new__(LLM4RecRecommender)
        recommender.processed_items=pd.DataFrame()
        recommender.processed_interactions=pd.DataFrame()
        recommender._processed_seen_items=Mock(return_value=set())
        recommender._load_processed_ranker=Mock(return_value=None)
        with self.assertRaises(FileNotFoundError):
            recommender._recommend_processed('u',None,10,100,[],False,'din',False,
                ['train','val'],'ranker_top_50',.7,.3,'none',{})

    def test_only_requested_head_is_sorted_and_history_feature_is_capped(self):
        recommender=object.__new__(LLM4RecRecommender)
        recommender.processed_items=pd.DataFrame([dict(item_id=str(i),categories=[],tags=[],author_id='',duration=0) for i in range(5)])
        recommender._processed_history_indices=Mock(return_value=list(range(1,11)))
        recommender._processed_item_popularity=Mock(return_value={})
        recommender._processed_user_activity=Mock(return_value={})
        encoder=SimpleNamespace(transform_one=lambda value:1,transform_list=lambda value,length:[])
        encoders=SimpleNamespace(user=encoder,item=encoder,author=encoder,category=encoder,tag=encoder,
                                 retrieval_source=encoder,duration_boundaries=[])
        captured={}
        def model(batch):
            captured.update(batch)
            return torch.tensor([0.,2.,1.,10.,11.])
        bundle={'model':model,'config':SimpleNamespace(max_history_len=3),'encoders':encoders}
        candidates=[Candidate(str(i),5-i,'trend',i+1) for i in range(5)]
        ranked=recommender._rerank_processed_candidates('u',candidates,bundle,5,'ranker_top_3',.7,.3,['train','val'])
        self.assertEqual([c.item_id for c in ranked],['1','2','0','3','4'])
        self.assertEqual(captured['history_item_indices'][0].tolist(),[8,9,10])
        self.assertEqual(captured['dense_features'][0,-1].item(),3)
        # Head scores describe ranker outputs, not an unused retrieval fusion.
        self.assertAlmostEqual(ranked[0].score, float(torch.sigmoid(torch.tensor(2.))), places=6)


if __name__=='__main__':unittest.main()
