import unittest

from llm4rec.evaluation.evaluator import _head_rerank_k, _ranker_head_order


class HeadRerankTest(unittest.TestCase):
    def test_ranker_top_mode_preserves_tail_order(self):
        self.assertEqual(_head_rerank_k("ranker_top_3"), 3)
        items = ["a", "b", "c", "d", "e"]
        scores = [0.1, 0.9, 0.2, 1.0, 0.0]
        self.assertEqual(_ranker_head_order(items, scores, output_k=5, head_k=3), ["b", "c", "a", "d", "e"])


if __name__ == "__main__":
    unittest.main()
