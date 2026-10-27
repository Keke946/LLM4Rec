import unittest

import pandas as pd

from llm4rec.datasets.mind import MINDAdapter


class MINDGroupSplitTest(unittest.TestCase):
    def test_one_impression_rows_stay_in_same_split(self):
        adapter = MINDAdapter(raw_dir=".")
        interactions = pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "N1",
                    "timestamp": 1,
                    "label": 1,
                    "watch_ratio": None,
                    "impression_id": "imp1",
                    "source": "mind",
                },
                {
                    "user_id": "u1",
                    "item_id": "N2",
                    "timestamp": 1,
                    "label": 0,
                    "watch_ratio": None,
                    "impression_id": "imp1",
                    "source": "mind",
                },
                {
                    "user_id": "u1",
                    "item_id": "N3",
                    "timestamp": 2,
                    "label": 1,
                    "watch_ratio": None,
                    "impression_id": "imp2",
                    "source": "mind",
                },
                {
                    "user_id": "u1",
                    "item_id": "N4",
                    "timestamp": 3,
                    "label": 1,
                    "watch_ratio": None,
                    "impression_id": "imp3",
                    "source": "mind",
                },
            ]
        )
        split = adapter.temporal_split(interactions)
        self.assertEqual(split.groupby("impression_id")["split"].nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
