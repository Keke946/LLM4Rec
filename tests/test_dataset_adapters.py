import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.datasets.movielens import MovieLensAdapter
from llm4rec.datasets.kuairec import KuaiRecAdapter


class DatasetAdapterTest(unittest.TestCase):
    def test_movielens_adapter_writes_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw"
            out = Path(tmpdir) / "processed"
            raw.mkdir()
            pd.DataFrame(
                [
                    {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation"},
                    {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children"},
                ]
            ).to_csv(raw / "movies.csv", index=False)
            pd.DataFrame(
                [
                    {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
                    {"userId": 1, "movieId": 2, "rating": 3.0, "timestamp": 2},
                    {"userId": 2, "movieId": 2, "rating": 4.5, "timestamp": 1},
                ]
            ).to_csv(raw / "ratings.csv", index=False)

            meta = MovieLensAdapter(raw).save_processed(out)

            self.assertTrue((out / "items.parquet").exists())
            self.assertTrue((out / "interactions.parquet").exists())
            self.assertTrue((out / "meta.json").exists())
            saved_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta.num_items, 2)
            self.assertEqual(saved_meta["num_interactions"], 3)

    def test_kuairec_adapter_threshold_and_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw"
            out = Path(tmpdir) / "processed"
            raw.mkdir()
            pd.DataFrame(
                [
                    {"video_id": 1, "category": "sports", "video_duration": 10},
                    {"video_id": 2, "category": "music", "video_duration": 20},
                ]
            ).to_csv(raw / "item_categories.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": "u1",
                        "video_id": 1,
                        "timestamp": 1,
                        "play_duration": 12,
                        "video_duration": 10,
                    },
                    {
                        "user_id": "u1",
                        "video_id": 2,
                        "timestamp": 2,
                        "play_duration": 5,
                        "video_duration": 20,
                    },
                ]
            ).to_csv(raw / "small_matrix.csv", index=False)

            KuaiRecAdapter(raw, positive_watch_ratio_threshold=1.0).save_processed(out)
            interactions = pd.read_parquet(out / "interactions.parquet")
            self.assertEqual(interactions["label"].tolist(), [1, 0])
            self.assertIn("watch_ratio", interactions.columns)


if __name__ == "__main__":
    unittest.main()
