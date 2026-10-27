from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.ranking.base import RankerConfig, masked_mean


class DeepFM(nn.Module):
    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim

        self.user_emb = nn.Embedding(config.num_users, dim, padding_idx=0)
        self.item_emb = nn.Embedding(config.num_items, dim, padding_idx=0)
        self.author_emb = nn.Embedding(config.num_authors, dim, padding_idx=0)
        self.category_emb = nn.Embedding(config.num_categories, dim, padding_idx=0)
        self.tag_emb = nn.Embedding(config.num_tags, dim, padding_idx=0)
        self.duration_emb = nn.Embedding(config.num_duration_buckets, dim, padding_idx=0)
        self.retrieval_source_emb = nn.Embedding(config.num_retrieval_sources, dim, padding_idx=0)

        self.user_linear = nn.Embedding(config.num_users, 1, padding_idx=0)
        self.item_linear = nn.Embedding(config.num_items, 1, padding_idx=0)
        self.author_linear = nn.Embedding(config.num_authors, 1, padding_idx=0)
        self.duration_linear = nn.Embedding(config.num_duration_buckets, 1, padding_idx=0)
        self.source_linear = nn.Embedding(config.num_retrieval_sources, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1))

        deep_layers: list[nn.Module] = []
        prev_dim = dim * 7 + config.dense_feature_dim
        for hidden_dim in config.hidden_dims:
            deep_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        field_vectors = self._field_vectors(batch)
        linear = (
            self.user_linear(batch["user_idx"])
            + self.item_linear(batch["item_idx"])
            + self.author_linear(batch["author_idx"])
            + self.duration_linear(batch["duration_bucket"])
            + self.source_linear(batch["retrieval_source_indices"]).sum(dim=1)
            + self.bias
        ).squeeze(-1)
        summed = field_vectors.sum(dim=1)
        fm = 0.5 * ((summed * summed) - (field_vectors * field_vectors).sum(dim=1)).sum(dim=1)
        dense_features = batch.get("dense_features")
        if dense_features is None:
            dense_features = torch.zeros(
                (batch["user_idx"].shape[0], self.config.dense_feature_dim),
                dtype=torch.float32,
                device=batch["user_idx"].device,
            )
        deep_input = torch.cat([field_vectors.flatten(start_dim=1), dense_features.float()], dim=-1)
        deep = self.deep(deep_input).squeeze(-1)
        return linear + fm + deep

    def _field_vectors(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        category_vec = masked_mean(self.category_emb(batch["category_indices"]), batch["category_indices"])
        tag_vec = masked_mean(self.tag_emb(batch["tag_indices"]), batch["tag_indices"])
        source_vec = masked_mean(
            self.retrieval_source_emb(batch["retrieval_source_indices"]),
            batch["retrieval_source_indices"],
        )
        return torch.stack(
            [
                self.user_emb(batch["user_idx"]),
                self.item_emb(batch["item_idx"]),
                self.author_emb(batch["author_idx"]),
                self.duration_emb(batch["duration_bucket"]),
                category_vec,
                tag_vec,
                source_vec,
            ],
            dim=1,
        )
