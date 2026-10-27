import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from llm4rec.evaluation.evaluator import OfflineEvaluator
from llm4rec.features import build_ranking_features
from llm4rec.ranking.lambdarank import (
    build_feature_schema,
    grouped_ltr_frame,
    make_lambdarank_features,
    make_relevance_labels,
    save_lambdarank_model,
)
from llm4rec.retrieval.base import Candidate


class DummyLambdaRankModel:
    def predict(self, features):
        return -features["rank"].to_numpy()


class LambdaRankTest(unittest.TestCase):
    def test_feature_schema_is_reused_for_train_and_predict(self):
        train = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i1", "label": 1, "rank": 1, "source_rank": 1, "retrieval_score": 0.9, "retrieval_sources": ["trend"], "item_popularity": 3, "is_injected_positive": 0},
                {"user_id": "u1", "item_id": "i2", "label": 0, "rank": 2, "source_rank": 2, "retrieval_score": 0.5, "retrieval_sources": ["trend"], "item_popularity": 2, "is_injected_positive": 0},
            ]
        )
        predict = train.drop(columns=["label"]).copy()
        schema = build_feature_schema(train)
        self.assertNotIn("label", schema["feature_columns"])
        self.assertNotIn("is_injected_positive", schema["feature_columns"])
        self.assertEqual(
            list(make_lambdarank_features(train, schema).columns),
            list(make_lambdarank_features(predict, schema).columns),
        )

    def test_group_sizes_are_built_by_user(self):
        frame = pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "i1", "label": 1, "rank": 1},
                {"user_id": "u1", "item_id": "i2", "label": 0, "rank": 2},
                {"user_id": "u2", "item_id": "i3", "label": 1, "rank": 1},
                {"user_id": "u2", "item_id": "i4", "label": 0, "rank": 2},
                {"user_id": "u3", "item_id": "i5", "label": 0, "rank": 1},
            ]
        )
        grouped, labels, groups = grouped_ltr_frame(
            frame,
            make_relevance_labels(frame),
            require_positive=True,
        )
        self.assertEqual(groups, [2, 2])
        self.assertEqual(len(grouped), 4)
        self.assertTrue(np.all(labels.reshape(2, 2)[:, 0] > 0))

    def test_evaluator_loads_lambdarank_and_reranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            features = root / "features"
            model_dir = root / "models" / "lambdarank"
            processed.mkdir()
            pd.DataFrame(
                [
                    {"item_id": "i1", "title": "A", "description": "", "categories": ["c1"], "tags": [], "author_id": "a1", "duration": 10, "full_text": "a"},
                    {"item_id": "i2", "title": "B", "description": "", "categories": ["c1"], "tags": [], "author_id": "a1", "duration": 20, "full_text": "b"},
                ]
            ).to_parquet(processed / "items.parquet", index=False)
            pd.DataFrame(
                [
                    {"user_id": "u1", "item_id": "i1", "timestamp": 1, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "train"},
                    {"user_id": "u1", "item_id": "i2", "timestamp": 2, "label": 1, "watch_ratio": 1.0, "impression_id": "", "source": "", "split": "test"},
                ]
            ).to_parquet(processed / "interactions.parquet", index=False)
            build_ranking_features(processed, features, max_history_len=3)
            schema = build_feature_schema(
                pd.DataFrame(
                    [
                        {"user_id": "u1", "item_id": "i1", "label": 1, "rank": 1, "source_rank": 1, "retrieval_score": 0.9, "retrieval_sources": ["trend"]},
                        {"user_id": "u1", "item_id": "i2", "label": 0, "rank": 2, "source_rank": 2, "retrieval_score": 0.1, "retrieval_sources": ["trend"]},
                    ]
                )
            )
            save_lambdarank_model(DummyLambdaRankModel(), model_dir, schema)
            (model_dir / "config.json").write_text('{"model": "lambdarank"}', encoding="utf-8")

            evaluator = OfflineEvaluator(processed, features_dir=features, ranker_root=root / "models")
            bundle = evaluator._load_ranker("lambdarank")
            self.assertIsNotNone(bundle)
            ranked = evaluator._rerank_with_ranker(
                "u1",
                [
                    Candidate(item_id="i2", score=0.1, source="trend", rank=2, retrieval_score=0.1),
                    Candidate(item_id="i1", score=0.9, source="trend", rank=1, retrieval_score=0.9),
                ],
                bundle,
                output_k=2,
                rerank_mode="ranker_top_2",
                retrieval_weight=0.7,
                ranker_weight=0.3,
                history_splits=["train"],
            )
            self.assertEqual(ranked, ["i1", "i2"])


if __name__ == "__main__":
    unittest.main()
