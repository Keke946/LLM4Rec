import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.retrieval import MetadataRetriever
from llm4rec.retrieval.base import BaseRetriever


class MetadataRetrieverTest(unittest.TestCase):
    def test_retrieves_tail_items_by_user_metadata_profile_and_loads(self):
        items = pd.DataFrame(
            [
                {"item_id": "i1", "categories": ["sports"], "tags": ["football"], "author_id": "a1", "full_text": "sports football"},
                {"item_id": "i2", "categories": ["sports"], "tags": ["basketball"], "author_id": "a2", "full_text": "sports basketball"},
                {"item_id": "i3", "categories": ["music"], "tags": ["piano"], "author_id": "a3", "full_text": "music piano"},
                {"item_id": "i4", "categories": ["sports"], "tags": ["running"], "author_id": "a4", "full_text": "sports running"},
            ]
        )
        train = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i1", "label": 1, "timestamp": 1, "split": "train"},
                {"user_id": "u2", "item_id": "i3", "label": 1, "timestamp": 1, "split": "train"},
            ]
        )
        retriever = MetadataRetriever(popularity_weight=0.0, tail_boost=0.0).fit(train, items)
        item_ids = [candidate.item_id for candidate in retriever.retrieve("u1", top_k=3)]

        self.assertNotIn("i1", item_ids)
        self.assertIn("i2", item_ids)
        self.assertIn("i4", item_ids)
        self.assertNotIn("i3", item_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metadata"
            retriever.save(path)
            loaded = BaseRetriever.load(path)
            loaded_ids = [candidate.item_id for candidate in loaded.retrieve("u1", top_k=3)]
            self.assertEqual(item_ids, loaded_ids)


if __name__ == "__main__":
    unittest.main()
