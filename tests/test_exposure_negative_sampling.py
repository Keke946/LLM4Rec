import unittest

import pandas as pd

from llm4rec.ranking.negative_sampling import sample_exposure_aware_negatives
from scripts.run_ltr_ablation import _metrics_markdown


class ExposureNegativeSamplingTest(unittest.TestCase):
    def test_future_test_positive_is_not_sampled_as_negative(self):
        candidates = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "pos", "label": 1, "rank": 1, "retrieval_source": "trend", "item_popularity": 5, "is_injected_positive": 0, "is_retrieved_positive": 1},
                {"user_id": "u1", "item_id": "future", "label": 0, "rank": 2, "retrieval_source": "trend", "item_popularity": 5, "is_injected_positive": 0, "is_retrieved_positive": 0},
                {"user_id": "u1", "item_id": "neg", "label": 0, "rank": 3, "retrieval_source": "trend", "item_popularity": 2, "is_injected_positive": 0, "is_retrieved_positive": 0},
            ]
        )
        interactions = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "future", "split": "test", "label": 1, "watch_ratio": 1.0},
                {"user_id": "u1", "item_id": "neg", "split": "train", "label": 0, "watch_ratio": 0.1},
            ]
        )
        with self.assertRaises(ValueError):
            sample_exposure_aware_negatives(candidates, interactions=interactions,
                                            future_positive_splits=["test"])
        sampled, stats = sample_exposure_aware_negatives(
            candidates, strategy="random_negative", negatives_per_positive=2)
        self.assertIn("future", set(sampled.item_id))
        self.assertEqual(stats["excluded_future_positive_negatives"], 0)

    def test_injected_positive_is_excluded_by_default(self):
        candidates = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "inj", "label": 1, "rank": 1, "retrieval_source": "", "item_popularity": 0, "is_injected_positive": 1, "is_retrieved_positive": 0},
                {"user_id": "u1", "item_id": "neg", "label": 0, "rank": 2, "retrieval_source": "trend", "item_popularity": 2, "is_injected_positive": 0, "is_retrieved_positive": 0},
            ]
        )
        sampled, stats = sample_exposure_aware_negatives(candidates, strategy="mixed_negative")
        self.assertNotIn("inj", set(sampled["item_id"]))
        self.assertEqual(stats["excluded_injected_positives"], 1)

    def test_metrics_markdown_generation(self):
        markdown = _metrics_markdown(
            pd.DataFrame(
                [
                    {
                        "experiment": "retrieval_only",
                        "status": "ok",
                        "Recall@10": 0.1,
                        "NDCG@10": 0.2,
                        "MRR@10": 0.3,
                    }
                ]
            )
        )
        self.assertIn("retrieval_only", markdown)
        self.assertIn("NDCG@10", markdown)


if __name__ == "__main__":
    unittest.main()
