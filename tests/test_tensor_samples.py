import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from llm4rec.features.build_features import RankingSampleDataset
from llm4rec.features.tensor_samples import TensorRankingDataset, TensorBatchLoader
from llm4rec.ranking.base import transform_dense


class TensorSamplesTest(unittest.TestCase):
    def test_cached_features_match_reference_and_exclude_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.parquet'
            rows = [dict(user_idx=1,item_idx=i+1,label=float(i==1),author_idx=0,duration_bucket=0,
                         category_indices=[1],tag_indices=[2],history_item_indices=[1,2],
                         retrieval_source_indices=[1],retrieval_score=.01,source_rank=i+1,
                         source_count=1,item_popularity=3,user_activity_level=5,history_len=2,
                         is_injected_positive=int(i==1)) for i in range(4)]
            pd.DataFrame(rows).to_parquet(path,index=False)
            reference=RankingSampleDataset(path,3)
            cache=TensorRankingDataset(path,3)
            for index in range(4):
                for key,value in cache.tensors.items():
                    torch.testing.assert_close(value[index], reference[index][key])
            excluded=TensorRankingDataset(path,3,exclude_injected=True)
            self.assertEqual(len(excluded),3)
            self.assertEqual(excluded.excluded_injected_rows,1)
            self.assertEqual(sum(len(b['label']) for b in TensorBatchLoader(excluded,2)),3)

    def test_log_transform_is_finite_and_backward_compatible(self):
        x=torch.tensor([[0.,1.,1500.,-1.]])
        torch.testing.assert_close(transform_dense(x,'none'),x)
        expected=torch.tensor([[0.,np.log(2),np.log(1501),0.]],dtype=torch.float32)
        torch.testing.assert_close(transform_dense(x,'log1p'),expected)


if __name__=='__main__':unittest.main()
