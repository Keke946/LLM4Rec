import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.datasets.movielens import MovieLensAdapter


class TemporalSplitTest(unittest.TestCase):
    def test_user_chronology_split_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir)
            pd.DataFrame(
                [
                    {"movieId": idx, "title": f"Movie {idx}", "genres": "Drama"}
                    for idx in range(1, 6)
                ]
            ).to_csv(raw / "movies.csv", index=False)
            pd.DataFrame(
                [
                    {"userId": 1, "movieId": idx, "rating": 5.0, "timestamp": idx}
                    for idx in range(1, 6)
                ]
            ).to_csv(raw / "ratings.csv", index=False)

            adapter = MovieLensAdapter(raw)
            interactions = adapter.temporal_split(adapter.build_interactions())
            train_max = interactions[interactions["split"] == "train"]["timestamp"].max()
            test_min = interactions[interactions["split"] == "test"]["timestamp"].min()

            self.assertLess(train_max, test_min)
            train_pairs = set(
                interactions[interactions["split"] == "train"][["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)
            )
            test_pairs = set(
                interactions[interactions["split"] == "test"][["user_id", "item_id", "timestamp"]].itertuples(index=False, name=None)
            )
            self.assertFalse(train_pairs & test_pairs)


if __name__ == "__main__":
    unittest.main()
