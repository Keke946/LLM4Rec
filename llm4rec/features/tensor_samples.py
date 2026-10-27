"""Cached tensor batches for candidate-level neural training and inference."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from llm4rec.features.build_features import _pad_list

SCALARS = ['user_idx', 'item_idx', 'author_idx', 'duration_bucket']
SEQUENCES = {'category_indices': 8, 'tag_indices': 16, 'retrieval_source_indices': 8}
DENSE = ['retrieval_score', 'source_rank', 'source_count', 'item_popularity',
         'user_activity_level', 'history_len']


def candidate_tensors(frame: pd.DataFrame, history_len: int) -> dict[str, torch.Tensor]:
    tensors = {}
    for name in SCALARS:
        values = pd.to_numeric(frame[name], errors='coerce').fillna(0) if name in frame else np.zeros(len(frame))
        tensors[name] = torch.from_numpy(np.asarray(values, dtype=np.int64).copy())
    for name, size in {**SEQUENCES, 'history_item_indices': history_len}.items():
        values = frame[name] if name in frame else [[] for _ in range(len(frame))]
        tensors[name] = torch.from_numpy(np.asarray([_pad_list(v, size) for v in values], dtype=np.int64).reshape(len(frame), size))
    dense = np.column_stack([pd.to_numeric(frame[name], errors='coerce').fillna(0)
                             if name in frame else np.zeros(len(frame)) for name in DENSE])
    tensors['dense_features'] = torch.from_numpy(dense.astype(np.float32))
    labels = pd.to_numeric(frame.label, errors='coerce').fillna(0) if 'label' in frame else np.zeros(len(frame))
    tensors['label'] = torch.from_numpy(np.asarray(labels, dtype=np.float32).copy())
    return tensors


class TensorRankingDataset:
    def __init__(self, path: str | Path, history_len=50, max_rows=None, seed=42,
                 exclude_injected=False):
        available = pq.read_schema(path).names
        wanted = SCALARS + list(SEQUENCES) + DENSE + ['history_item_indices', 'label', 'is_injected_positive']
        frame = pd.read_parquet(path, columns=[name for name in dict.fromkeys(wanted) if name in available])
        self.input_rows = len(frame)
        self.excluded_injected_rows = 0
        if exclude_injected:
            if 'is_injected_positive' not in frame:
                raise ValueError('Audited candidate flags are required to exclude injected rows.')
            self.excluded_injected_rows = int((frame.is_injected_positive > 0).sum())
            frame = frame[frame.is_injected_positive == 0]
        self.eligible_rows = len(frame)
        if max_rows and max_rows < len(frame):
            frame = frame.sample(n=max_rows, random_state=seed)
        self.tensors = candidate_tensors(frame.reset_index(drop=True), history_len)

    def __len__(self):
        return len(self.tensors['label'])


class TensorBatchLoader:
    def __init__(self, dataset, batch_size, shuffle=False, generator=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.generator = shuffle, generator

    def __iter__(self):
        indices = (torch.randperm(len(self.dataset), generator=self.generator)
                   if self.shuffle else torch.arange(len(self.dataset)))
        for selected in indices.split(self.batch_size):
            yield {name: value.index_select(0, selected) for name, value in self.dataset.tensors.items()}
