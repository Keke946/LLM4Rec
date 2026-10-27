import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm4rec.config import ProjectPaths
from llm4rec.recommender import LLM4RecRecommender


class RecommenderTest(unittest.TestCase):
    def test_offline_recommender_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ProjectPaths.from_root(tmpdir)
            paths.data_dir.mkdir(parents=True)
            paths.models_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "video_id": "vid_001",
                        "title": "Funny Cat Compilation",
                        "description": "A funny cat video.",
                        "tags": "['funny', 'cat', 'compilation']",
                        "labels": "['Comedy']",
                    },
                    {
                        "video_id": "vid_002",
                        "title": "Football Goals",
                        "description": "Sports highlights.",
                        "tags": "['football', 'sports', 'goals']",
                        "labels": "['Sports']",
                    },
                ]
            ).to_csv(paths.video_csv, index=False)

            recommender = LLM4RecRecommender(paths=paths, allow_vector_recall=False)
            results = recommender.recommend("猫猫搞笑视频", k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["video_id"], "vid_001")
            self.assertGreater(results[0]["score"], 0)
            self.assertIn("reason", results[0])

    def test_llm_hallucinated_item_id_is_rejected_in_recommender_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ProjectPaths.from_root(tmpdir)
            paths.data_dir.mkdir(parents=True)
            paths.models_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "video_id": "vid_001",
                        "title": "Funny Cat Compilation",
                        "description": "A funny cat video.",
                        "tags": "['funny', 'cat']",
                        "labels": "['Comedy']",
                    }
                ]
            ).to_csv(paths.video_csv, index=False)

            recommender = LLM4RecRecommender(
                paths=paths,
                allow_vector_recall=False,
                use_llm_explanations=True,
            )
            with patch(
                "llm4rec.explainer.llm_explanation",
                return_value="推荐 item_999，因为它有 99999 次播放。",
            ):
                results = recommender.recommend("space opera", k=1)

            self.assertEqual(len(results), 1)
            self.assertNotIn("item_999", results[0]["reason"])
            self.assertNotIn("99999", results[0]["reason"])


if __name__ == "__main__":
    unittest.main()
