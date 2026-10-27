import unittest

from llm4rec.retrieval.base import Candidate
from llm4rec.retrieval.merge import reciprocal_rank_fusion


class WeightedFusionTest(unittest.TestCase):
    def test_source_weights_change_rrf_order(self):
        popular = [Candidate(item_id="head", score=1.0, source="popular", rank=1)]
        metadata = [Candidate(item_id="tail", score=1.0, source="metadata", rank=1)]

        default_order = reciprocal_rank_fusion([popular, metadata], top_k=2)
        weighted_order = reciprocal_rank_fusion(
            [popular, metadata],
            top_k=2,
            weights={"metadata": 2.0, "popular": 0.1},
        )

        self.assertEqual(default_order[0].item_id, "head")
        self.assertEqual(weighted_order[0].item_id, "tail")


if __name__ == "__main__":
    unittest.main()
