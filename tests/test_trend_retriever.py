import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.retrieval import TrendRetriever
from llm4rec.retrieval.base import BaseRetriever


class TrendRetrieverTest(unittest.TestCase):
    def test_recent_positive_beats_old_popularity_and_loads(self):
        items = pd.DataFrame(
            [
                {"item_id": "old", "categories": ["c"], "tags": ["t"], "full_text": "old c"},
                {"item_id": "recent", "categories": ["c"], "tags": ["t"], "full_text": "recent c"},
                {"item_id": "other", "categories": ["x"], "tags": ["x"], "full_text": "other"},
            ]
        )
        train = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "old", "label": 1, "timestamp": 1, "split": "train"},
                {"user_id": "u2", "item_id": "old", "label": 1, "timestamp": 1, "split": "train"},
                {"user_id": "u3", "item_id": "old", "label": 1, "timestamp": 1, "split": "train"},
                {"user_id": "u4", "item_id": "recent", "label": 1, "timestamp": 100000, "split": "train"},
            ]
        )

        retriever = TrendRetriever(half_life_days=0.01, category_weight=0.0, tag_weight=0.0).fit(train, items)
        recs = [candidate.item_id for candidate in retriever.retrieve("cold_user", top_k=3)]
        self.assertLess(recs.index("recent"), recs.index("old"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trend"
            retriever.save(path)
            loaded = BaseRetriever.load(path)
            loaded_recs = [candidate.item_id for candidate in loaded.retrieve("cold_user", top_k=3)]
            self.assertEqual(recs, loaded_recs)


if __name__ == "__main__":
    unittest.main()
