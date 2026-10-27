from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer


class CTRMLP(nn.Module):
    """Simple MLP ranker for user-tag/video-tag CTR prediction."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (128, 64)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def binarize_tags(tags: Sequence[str], binarizer: MultiLabelBinarizer) -> list[int]:
    class_to_index = {tag: idx for idx, tag in enumerate(binarizer.classes_)}
    vector = [0] * len(binarizer.classes_)
    for tag in tags:
        idx = class_to_index.get(str(tag).lower())
        if idx is not None:
            vector[idx] = 1
    return vector


def build_ctr_feature_matrix(
    user_tags: Sequence[str],
    video_tags: Sequence[Sequence[str]],
    binarizer: MultiLabelBinarizer,
) -> np.ndarray:
    user_vec = binarize_tags(user_tags, binarizer)
    rows = [user_vec + binarize_tags(tags, binarizer) for tags in video_tags]
    return np.asarray(rows, dtype=np.float32)
