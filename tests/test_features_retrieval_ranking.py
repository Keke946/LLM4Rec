import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from llm4rec.datasets.movielens import MovieLensAdapter
from llm4rec.explainer import reject_hallucinated_item_ids
from llm4rec.features import FeatureEncoders, RankingSampleDataset, build_ranking_features
from llm4rec.ranking import DeepFM, DIN, EmbeddingMLP
from llm4rec.ranking.base import RankerConfig
from llm4rec.retrieval import (
    BPRMFRetriever,
    ItemCFRetriever,
    LightGCNRetriever,
    MetadataRetriever,
    PopularRetriever,
    SASRecRetriever,
    TwoTowerRetriever,
)


class FeatureRetrievalRankingTest(unittest.TestCase):
    def test_features_retrievers_rankers_and_llm_guardrail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw"
            processed = root / "processed"
            features = root / "features"
            raw.mkdir()
            pd.DataFrame(
                [
                    {"movieId": 1, "title": "Action One", "genres": "Action"},
                    {"movieId": 2, "title": "Action Two", "genres": "Action"},
                    {"movieId": 3, "title": "Drama One", "genres": "Drama"},
                    {"movieId": 4, "title": "Comedy One", "genres": "Comedy"},
                ]
            ).to_csv(raw / "movies.csv", index=False)
            pd.DataFrame(
                [
                    {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
                    {"userId": 1, "movieId": 2, "rating": 5.0, "timestamp": 2},
                    {"userId": 1, "movieId": 3, "rating": 4.0, "timestamp": 3},
                    {"userId": 2, "movieId": 1, "rating": 5.0, "timestamp": 1},
                    {"userId": 2, "movieId": 4, "rating": 5.0, "timestamp": 2},
                    {"userId": 2, "movieId": 3, "rating": 3.0, "timestamp": 3},
                ]
            ).to_csv(raw / "ratings.csv", index=False)

            MovieLensAdapter(raw).save_processed(processed)
            build_ranking_features(processed, features, max_history_len=3)
            encoders = FeatureEncoders.load(features / "encoders.pkl")
            self.assertEqual(encoders.item.transform_one("unknown"), 0)

            dataset = RankingSampleDataset(features / "train_rank_samples.parquet", max_history_len=3)
            sample = dataset[0]
            self.assertEqual(sample["history_item_indices"].shape[0], 3)

            items = pd.read_parquet(processed / "items.parquet")
            interactions = pd.read_parquet(processed / "interactions.parquet")
            train = interactions[interactions["split"] == "train"]
            for retriever in [
                PopularRetriever(),
                ItemCFRetriever(),
                BPRMFRetriever(epochs=1, embedding_dim=4),
                TwoTowerRetriever(epochs=1, embedding_dim=4, max_history_len=3),
                LightGCNRetriever(epochs=1, embedding_dim=4, num_layers=1),
                SASRecRetriever(epochs=1, embedding_dim=4, max_seq_len=3, num_heads=2, num_layers=1),
                MetadataRetriever(),
            ]:
                retriever.fit(train, items)
                candidates = retriever.retrieve("1", top_k=10)
                train_seen = set(train[(train["user_id"] == "1") & (train["label"] == 1)]["item_id"])
                self.assertFalse({candidate.item_id for candidate in candidates} & train_seen)

            config = RankerConfig(
                num_users=len(encoders.user),
                num_items=len(encoders.item),
                num_authors=len(encoders.author),
                num_categories=len(encoders.category),
                num_tags=len(encoders.tag),
                num_retrieval_sources=len(encoders.retrieval_source),
                num_duration_buckets=len(encoders.duration_boundaries) + 2,
                embedding_dim=8,
                max_history_len=3,
            )
            batch = {key: value.unsqueeze(0) for key, value in sample.items() if key != "label"}
            for model_cls in [EmbeddingMLP, DeepFM, DIN]:
                logits = model_cls(config)(batch)
                self.assertEqual(tuple(logits.shape), (1,))
                self.assertFalse(torch.isnan(logits).any())

            guarded = reject_hallucinated_item_ids("推荐 item_999，因为它很好", ["item_1"], "fallback")
            self.assertEqual(guarded, "fallback")


if __name__ == "__main__":
    unittest.main()
