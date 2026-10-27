from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.ranking.base import RankerConfig


class CTRMLPRanker(nn.Module):
    """Baseline ranker using normalized integer IDs as dense features."""

    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = 6
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.stack(
            [
                batch["user_idx"].float() / max(1, self.config.num_users),
                batch["item_idx"].float() / max(1, self.config.num_items),
                batch["author_idx"].float() / max(1, self.config.num_authors),
                batch["duration_bucket"].float() / max(1, self.config.num_duration_buckets),
                (batch["category_indices"] != 0).float().sum(dim=1) / 8.0,
                (batch["tag_indices"] != 0).float().sum(dim=1) / 16.0,
            ],
            dim=1,
        )
        return self.mlp(features).squeeze(-1)
