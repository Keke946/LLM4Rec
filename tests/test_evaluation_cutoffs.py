import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.evaluation.evaluator import OfflineEvaluator
from llm4rec.retrieval import PopularRetriever


class EvaluationCutoffTest(unittest.TestCase):
    def test_large_cutoffs_use_candidate_pool_not_final_top10(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
            processed.mkdir()
            item_ids = [f"i{idx}" for idx in range(1, 61)]
            pd.DataFrame(
                [
                    {
                        "item_id": item_id,
                        "title": item_id,
                        "description": "",
                        "categories": ["cat"],
                        "tags": [],
                        "author_id": "",
                        "duration": None,
                        "full_text": item_id,
                    }
                    for item_id in item_ids
                ]
            ).to_parquet(processed / "items.parquet", index=False)

            rows = []
            timestamp = 1
            for rank, item_id in enumerate(item_ids, start=1):
                # Popularity creates a deterministic global ranking: i1, i2, ... i60.
                for user_idx in range(61 - rank):
                    rows.append(
                        {
                            "user_id": f"pop_{rank}_{user_idx}",
                            "item_id": item_id,
                            "timestamp": timestamp,
                            "label": 1,
                            "watch_ratio": None,
                            "impression_id": "",
                            "source": "test",
                            "split": "train",
                        }
                    )
                    timestamp += 1
            rows.append(
                {
                    "user_id": "target",
                    "item_id": "i30",
                    "timestamp": timestamp,
                    "label": 1,
                    "watch_ratio": None,
                    "impression_id": "",
                    "source": "test",
                    "split": "test",
                }
            )
            pd.DataFrame(rows).to_parquet(processed / "interactions.parquet", index=False)

            result = OfflineEvaluator(processed).evaluate_retrievers(
                ["popular"],
                candidate_pool_k=60,
                cutoffs=[10, 50],
                final_top_k=10,
                filter_seen_splits=["train"],
            )

            summary = result["retrieval_only_summary"]
            self.assertEqual(summary["Recall@10"], 0.0)
            self.assertEqual(summary["Recall@50"], 1.0)

    def test_train_and_val_positives_are_filtered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir) / "processed"
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
                    for idx in range(1, 6)
                ]
            ).to_parquet(processed / "items.parquet", index=False)
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "user_id": f"pop{idx}",
                        "item_id": "i1",
                        "timestamp": idx,
                        "label": 1,
                        "watch_ratio": None,
                        "impression_id": "",
                        "source": "test",
                        "split": "train",
                    }
                )
            rows.extend(
                [
                    {
                        "user_id": "target",
                        "item_id": "i1",
                        "timestamp": 20,
                        "label": 1,
                        "watch_ratio": None,
                        "impression_id": "",
                        "source": "test",
                        "split": "val",
                    },
                    {
                        "user_id": "target",
                        "item_id": "i2",
                        "timestamp": 30,
                        "label": 1,
                        "watch_ratio": None,
                        "impression_id": "",
                        "source": "test",
                        "split": "test",
                    },
                ]
            )
            pd.DataFrame(rows).to_parquet(processed / "interactions.parquet", index=False)
            evaluator = OfflineEvaluator(processed)
            retriever = PopularRetriever().fit(evaluator.train, evaluator.items)
            excluded = evaluator._positive_items_by_user(["train", "val"])["target"]
            candidates = evaluator._filter_candidate_objects(
                retriever.retrieve("target", top_k=5),
                excluded,
            )
            self.assertNotIn("i1", [candidate.item_id for candidate in candidates])


if __name__ == "__main__":
    unittest.main()
