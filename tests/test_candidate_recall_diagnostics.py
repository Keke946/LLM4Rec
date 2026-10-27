import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.diagnose_candidate_recall import diagnose


class CandidateRecallDiagnosticsTest(unittest.TestCase):
    def test_empty_live_pool_is_a_valid_diagnostic(self):
        samples = pd.DataFrame([dict(user_id="u", item_id="i", label=1,
                                    is_retrieved_positive=0, is_injected_positive=1,
                                    retrieval_sources=[], history_len=0, item_popularity=0)])
        summary, _, _, sources = diagnose(samples, None, "test")
        self.assertEqual(summary["candidate_positive_recall"], 0)
        self.assertEqual(list(sources.columns)[0], "source")
        self.assertTrue(sources.empty)

    def test_diagnose_reports_buckets_and_source_credit(self):
        samples = pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "label": 1,
                    "is_retrieved_positive": 1,
                    "is_injected_positive": 0,
                    "retrieval_sources": ["metadata", "popular"],
                    "item_popularity": 0,
                    "history_len": 10,
                },
                {
                    "user_id": "u1",
                    "item_id": "i2",
                    "label": 1,
                    "is_retrieved_positive": 0,
                    "is_injected_positive": 1,
                    "retrieval_sources": [],
                    "item_popularity": 50,
                    "history_len": 10,
                },
                {
                    "user_id": "u1",
                    "item_id": "i3",
                    "label": 0,
                    "is_retrieved_positive": 0,
                    "is_injected_positive": 0,
                    "retrieval_sources": ["metadata"],
                    "item_popularity": 5,
                    "history_len": 10,
                },
            ]
        )
        interactions = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i1", "label": 1, "split": "train"},
                {"user_id": "u1", "item_id": "i2", "label": 1, "split": "test"},
            ]
        )

        summary, popularity, history, source = diagnose(samples, interactions, "test")

        self.assertEqual(summary["positives"], 2)
        self.assertEqual(summary["retrieved_positives"], 1)
        self.assertAlmostEqual(summary["candidate_positive_recall"], 0.5)
        self.assertIn("cold_0", set(popularity["item_popularity_bucket"].astype(str)))
        self.assertIn("h6_20", set(history["history_len_bucket"].astype(str)))
        metadata_row = source[source["source"] == "metadata"].iloc[0]
        self.assertEqual(int(metadata_row["retrieved_positive_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
