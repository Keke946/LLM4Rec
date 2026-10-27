import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.features import build_ranking_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RankerCandidateBuilderTest(unittest.TestCase):
    def test_builds_retriever_backed_candidate_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            features = root / "features"
            retrievers = root / "retrievers"
            processed.mkdir()
            pd.DataFrame(
                [
                    {"item_id": "i1", "title": "A", "categories": ["c1"], "tags": ["t1"], "author_id": "a1", "duration": 10, "full_text": "a c1 t1"},
                    {"item_id": "i2", "title": "B", "categories": ["c1"], "tags": ["t2"], "author_id": "a1", "duration": 20, "full_text": "b c1 t2"},
                    {"item_id": "i3", "title": "C", "categories": ["c2"], "tags": ["t3"], "author_id": "a2", "duration": 30, "full_text": "c c2 t3"},
                    {"item_id": "i4", "title": "D", "categories": ["c2"], "tags": ["t4"], "author_id": "a2", "duration": 40, "full_text": "d c2 t4"},
                    {"item_id": "i5", "title": "E", "categories": ["c3"], "tags": ["t5"], "author_id": "a3", "duration": 50, "full_text": "e c3 t5"},
                ]
            ).to_parquet(processed / "items.parquet", index=False)
            pd.DataFrame(
                [
                    {"user_id": "u1", "item_id": "i1", "timestamp": 1, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "train"},
                    {"user_id": "u1", "item_id": "i2", "timestamp": 2, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "val"},
                    {"user_id": "u1", "item_id": "i3", "timestamp": 3, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "test"},
                    {"user_id": "u2", "item_id": "i2", "timestamp": 1, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "train"},
                    {"user_id": "u2", "item_id": "i4", "timestamp": 2, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "val"},
                    {"user_id": "u2", "item_id": "i5", "timestamp": 3, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "test"},
                ]
            ).to_parquet(processed / "interactions.parquet", index=False)
            build_ranking_features(processed, features, max_history_len=3)

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_ranker_candidates.py"),
                    "--dataset",
                    "toy",
                    "--processed_dir",
                    str(processed),
                    "--features_dir",
                    str(features),
                    "--retriever_dir",
                    str(retrievers),
                    "--retrievers",
                    "popular,itemcf",
                    "--candidate_pool_k",
                    "4",
                    "--max_history_len",
                    "3",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )

            train_samples = pd.read_parquet(features / "train_rank_samples.parquet")
            self.assertGreater(len(train_samples), 0)
            self.assertGreater(train_samples["label"].sum(), 0)
            for column in [
                "retrieval_score",
                "retrieval_source",
                "rank",
                "source_count",
                "source_rank",
                "is_injected_positive",
                "is_retrieved_positive",
            ]:
                self.assertIn(column, train_samples.columns)
            self.assertFalse(train_samples["retrieval_source"].astype(str).str.contains("label_positive").any())
            injected = train_samples[train_samples["is_injected_positive"] == 1]
            if not injected.empty:
                self.assertTrue((injected["source_rank"] == 0).all())
            for _, group in train_samples.groupby("user_id"):
                live_count = int((group.is_injected_positive == 0).sum())
                self.assertTrue((group.candidate_pool_size == live_count).all())
            meta = json.loads((features / "ranker_candidate_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["retrievers"], ["popular", "itemcf"])
            self.assertIn("positive_injection_rate", meta["splits"]["train"])
            self.assertIn("candidate_positive_recall", meta["overall"])


if __name__ == "__main__":
    unittest.main()
