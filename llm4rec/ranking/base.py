from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankerConfig:
    num_users: int
    num_items: int
    num_authors: int
    num_categories: int
    num_tags: int
    num_retrieval_sources: int = 9
    num_duration_buckets: int = 9
    dense_feature_dim: int = 6
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 64)
    max_history_len: int = 20
    dense_transform: str = "none"


def masked_mean(embeddings, indices):
    mask = (indices != 0).float().unsqueeze(-1)
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def transform_dense(values, mode):
    if mode == "none":
        return values.float()
    if mode == "log1p":
        import torch
        return torch.log1p(values.float().clamp_min(0))
    raise ValueError(f"Unknown dense transform: {mode}")
