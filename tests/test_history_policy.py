import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.features import build_ranking_features


class HistoryPolicyTest(unittest.TestCase):
    def test_val_and_test_history_use_only_past_positive_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            features = root / "features"
            processed.mkdir()
            pd.DataFrame(
                [
                    {
                        "item_id": f"i{idx}",
                        "title": f"i{idx}",
                        "description": "",
                        "categories": ["cat"],
                        "tags": [],
                        "author_id": "",
                        "duration": None,
                        "full_text": f"i{idx}",
                    }
                    for idx in range(1, 5)
                ]
            ).to_parquet(processed / "items.parquet", index=False)
            pd.DataFrame(
                [
                    {"user_id": "u1", "item_id": "i1", "timestamp": 1, "label": 1, "watch_ratio": None, "impression_id": "", "source": "x", "split": "train"},
                    {"user_id": "u1", "item_id": "i2", "timestamp": 2, "label": 1, "watch_ratio": None, "impression_id": "", "source": "x", "split": "val"},
                    {"user_id": "u1", "item_id": "i3", "timestamp": 3, "label": 1, "watch_ratio": None, "impression_id": "", "source": "x", "split": "test"},
                ]
            ).to_parquet(processed / "interactions.parquet", index=False)

            encoders = build_ranking_features(processed, features, max_history_len=5)
            val = pd.read_parquet(features / "val_rank_samples.parquet")
            test = pd.read_parquet(features / "test_rank_samples.parquet")

            train_item = encoders.item.transform_one("i1")
            val_item = encoders.item.transform_one("i2")
            self.assertEqual(list(val.iloc[0]["history_item_indices"]), [train_item])
            self.assertEqual(list(test.iloc[0]["history_item_indices"]), [train_item, val_item])


if __name__ == "__main__":
    unittest.main()
