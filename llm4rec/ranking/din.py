from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.ranking.base import RankerConfig, masked_mean


class DIN(nn.Module):
    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.user_emb = nn.Embedding(config.num_users, dim, padding_idx=0)
        self.item_emb = nn.Embedding(config.num_items, dim, padding_idx=0)
        self.category_emb = nn.Embedding(config.num_categories, dim, padding_idx=0)
        self.tag_emb = nn.Embedding(config.num_tags, dim, padding_idx=0)
        self.author_emb = nn.Embedding(config.num_authors, dim, padding_idx=0)
        self.duration_emb = nn.Embedding(config.num_duration_buckets, dim, padding_idx=0)
        self.retrieval_source_emb = nn.Embedding(config.num_retrieval_sources, dim, padding_idx=0)

        self.attention = nn.Sequential(
            nn.Linear(dim * 4, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

        input_dim = dim * 8 + config.dense_feature_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_vec = self.item_emb(batch["item_idx"])
        history_vecs = self.item_emb(batch["history_item_indices"])
        mask = batch["history_item_indices"] != 0
        repeated_item = item_vec.unsqueeze(1).expand_as(history_vecs)
        attention_input = torch.cat(
            [
                history_vecs,
                repeated_item,
                history_vecs - repeated_item,
                history_vecs * repeated_item,
            ],
            dim=-1,
        )
        attention_scores = self.attention(attention_input).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        history_interest = (history_vecs * weights).sum(dim=1)
        empty_history = (~mask).all(dim=1, keepdim=True)
        history_interest = torch.where(empty_history, torch.zeros_like(history_interest), history_interest)

        category_vec = masked_mean(self.category_emb(batch["category_indices"]), batch["category_indices"])
        tag_vec = masked_mean(self.tag_emb(batch["tag_indices"]), batch["tag_indices"])
        source_vec = masked_mean(
            self.retrieval_source_emb(batch["retrieval_source_indices"]),
            batch["retrieval_source_indices"],
        )
        dense_features = batch.get("dense_features")
        if dense_features is None:
            dense_features = torch.zeros(
                (batch["user_idx"].shape[0], self.config.dense_feature_dim),
                dtype=torch.float32,
                device=batch["user_idx"].device,
            )
        features = torch.cat(
            [
                self.user_emb(batch["user_idx"]),
                item_vec,
                history_interest,
                self.author_emb(batch["author_idx"]),
                self.duration_emb(batch["duration_bucket"]),
                category_vec,
                tag_vec,
                source_vec,
                dense_features.float(),
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)
