import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.retrieval import ExposureQualityRetriever
from llm4rec.retrieval.base import BaseRetriever


class ExposureQualityRetrieverTest(unittest.TestCase):
    def test_high_watch_ratio_recent_item_wins_and_loads(self):
        items = pd.DataFrame(
            [
                {"item_id": "low_recent"},
                {"item_id": "high_recent"},
                {"item_id": "high_old"},
            ]
        )
        train = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "low_recent", "label": 0, "watch_ratio": 0.1, "timestamp": 1000},
                {"user_id": "u2", "item_id": "high_recent", "label": 0, "watch_ratio": 0.95, "timestamp": 1000},
                {"user_id": "u3", "item_id": "high_old", "label": 0, "watch_ratio": 1.0, "timestamp": 1},
            ]
        )
        retriever = ExposureQualityRetriever(half_life_days=0.01).fit(train, items)
        recs = [candidate.item_id for candidate in retriever.retrieve("cold_user", top_k=3)]
        self.assertEqual(recs[0], "high_recent")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exposure"
            retriever.save(path)
            loaded = BaseRetriever.load(path)
            loaded_recs = [candidate.item_id for candidate in loaded.retrieve("cold_user", top_k=3)]
            self.assertEqual(recs, loaded_recs)


if __name__ == "__main__":
    unittest.main()
