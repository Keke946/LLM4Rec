import unittest

from llm4rec.scoring import compute_final_scores


class RerankScoringTest(unittest.TestCase):
    def test_fusion_does_not_blindly_destroy_retrieval_ranking(self):
        retrieval_scores = [1.0, 0.9, 0.1]
        ranker_logits = [-10.0, -10.0, 10.0]
        final_scores = compute_final_scores(
            retrieval_scores,
            ranker_logits,
            mode="retrieval_ranker_fusion",
            retrieval_weight=0.9,
            ranker_weight=0.1,
        )
        self.assertEqual(max(range(len(final_scores)), key=lambda idx: final_scores[idx]), 0)

    def test_ranker_only_can_choose_ranker_favorite(self):
        retrieval_scores = [1.0, 0.9, 0.1]
        ranker_logits = [-10.0, -10.0, 10.0]
        final_scores = compute_final_scores(retrieval_scores, ranker_logits, mode="ranker_only")
        self.assertEqual(max(range(len(final_scores)), key=lambda idx: final_scores[idx]), 2)


if __name__ == "__main__":
    unittest.main()
