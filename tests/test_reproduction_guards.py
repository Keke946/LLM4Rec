import unittest

import numpy as np
import pandas as pd

from llm4rec.datasets.kuairec import KuaiRecAdapter
from llm4rec.evaluation.metrics import ranking_metrics
from llm4rec.ranking.gbdt_features import gbdt_feature_row, make_gbdt_features
from llm4rec.validation import validate_training_splits


class ReproductionGuardsTest(unittest.TestCase):
    def test_test_split_cannot_be_used_for_training_or_validation(self):
        validate_training_splits("train", "val")
        for train, val in [("val", "test"), ("test", "val"), ("train", "train")]:
            with self.assertRaises(ValueError):
                validate_training_splits(train, val)

    def test_explicit_big_matrix_never_falls_back_to_small(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"small_matrix.csv").write_text("user_id,video_id,watch_ratio\n0,0,1\n")
            with self.assertRaises(FileNotFoundError):
                KuaiRecAdapter(directory, kuairec_matrix="big").load_raw()

    def test_global_split_has_no_cross_user_time_overlap(self):
        frame = pd.DataFrame({"user_id": ["u1"] * 20 + ["u2"] * 20,
                              "item_id": list(map(str, range(40))),
                              "timestamp": list(range(40))})
        result = KuaiRecAdapter(".", split_strategy="global_temporal").temporal_split(frame)
        train, val, test = [result[result.split == s] for s in ["train", "val", "test"]]
        self.assertLess(train.timestamp.max(), val.timestamp.min())
        self.assertLess(val.timestamp.max(), test.timestamp.min())

    def test_metrics_hand_calculation_and_duplicate_guard(self):
        m = ranking_metrics(["x", "a", "b"], {"a", "b", "c"}, [3, 10])
        self.assertAlmostEqual(m["Recall@3"], 2 / 3)
        self.assertAlmostEqual(m["MRR@10"], 1 / 2)
        self.assertAlmostEqual(m["MAP@10"], (1 / 2 + 2 / 3) / 3)
        ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
        self.assertAlmostEqual(m["NDCG@3"], (1 / np.log2(3) + 1 / np.log2(4)) / ideal)
        short = ranking_metrics(["a"], {"a", "b", "c"}, [10])
        self.assertAlmostEqual(short["MAP@10"], 1 / 3)
        with self.assertRaises(ValueError):
            ranking_metrics(["a", "a"], {"a"}, [10])

    def test_dense_negative_sampler_never_returns_a_positive(self):
        from random import Random
        from llm4rec.retrieval.sampling import sample_negative
        rng = Random(42)
        for _ in range(50):
            self.assertEqual(sample_negative(rng, range(100), set(range(99))), 99)
        with self.assertRaises(ValueError):
            sample_negative(rng, range(2), {0, 1})

    def test_vectorized_features_match_online_row(self):
        rows = [{"rank": 2, "source_rank": 2, "retrieval_sources": ["trend"],
                 "item_popularity": 5, "category_indices": [0, 2, 3],
                 "candidate_pool_size": 1500, "history_len": 50}, {}]
        expected = pd.DataFrame([gbdt_feature_row(r, ["trend"]) for r in rows])
        actual = make_gbdt_features(pd.DataFrame(rows), ["trend"])
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
