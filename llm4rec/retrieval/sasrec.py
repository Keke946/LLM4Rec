from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm4rec.retrieval.base import BaseRetriever, Candidate

from llm4rec.retrieval.sampling import sample_negative


class SASRecDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[int, list[int], int]],
        positives_by_user: dict[int, set[int]],
        num_items: int,
        max_seq_len: int,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.positives_by_user = positives_by_user
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, seq, pos_idx = self.samples[index]
        positives = self.positives_by_user.get(user_idx, set())
        neg_idx = sample_negative(self.rng, range(1, self.num_items + 1), positives)
        return (
            torch.tensor(_pad_left(seq, self.max_seq_len), dtype=torch.long),
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


class _SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.max_seq_len = max_seq_len
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def sequence_vector(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0).expand_as(seq)
        x = self.item_emb(seq) + self.pos_emb(positions)
        padding_mask = seq == 0
        causal_mask = torch.triu(
            torch.ones(seq.shape[1], seq.shape[1], device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)
        lengths = (seq != 0).sum(dim=1).clamp_min(1) - 1
        return encoded[torch.arange(seq.shape[0], device=seq.device), lengths]

    def score(self, seq: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        seq_vec = self.sequence_vector(seq)
        item_vec = self.item_emb(items)
        return (seq_vec * item_vec).sum(dim=-1)


class SASRecRetriever(BaseRetriever):
    name = "sasrec"

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        lr: float = 1e-3,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 42,
        batch_size: int = 1024,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "SASRecRetriever":
        super().fit(train_interactions, items)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        users = sorted(train_interactions["user_id"].astype(str).unique())
        item_ids = sorted(items["item_id"].astype(str).unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}
        self.idx_to_item = {idx + 1: item_id for idx, item_id in enumerate(item_ids)}
        positives = train_interactions[train_interactions["label"] == 1].sort_values(["user_id", "timestamp", "item_id"])

        self.histories: dict[str, list[int]] = {}
        samples: list[tuple[int, list[int], int]] = []
        positives_by_user_idx: dict[int, set[int]] = {}
        for user_id, group in positives.groupby("user_id"):
            sequence = [
                self.item_to_idx[str(item_id)]
                for item_id in group["item_id"].astype(str)
                if str(item_id) in self.item_to_idx
            ]
            user_key = str(user_id)
            self.histories[user_key] = sequence[-self.max_seq_len :]
            if user_key not in self.user_to_idx:
                continue
            user_idx = self.user_to_idx[user_key]
            positives_by_user_idx[user_idx] = set(sequence)
            for pos in range(1, len(sequence)):
                samples.append((user_idx, sequence[:pos], sequence[pos]))

        self.model = _SASRec(
            max(1, len(item_ids)),
            self.embedding_dim,
            self.max_seq_len,
            self.num_heads,
            self.num_layers,
            self.dropout,
        ).to(self.device)
        if not samples:
            self.model.to("cpu")
            self.device = "cpu"
            return self

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = SASRecDataset(samples, positives_by_user_idx, len(item_ids), self.max_seq_len, seed=self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            self.model.train()
            for seq_t, _, pos_t, neg_t in loader:
                seq_t = seq_t.to(self.device)
                pos_t = pos_t.to(self.device)
                neg_t = neg_t.to(self.device)
                pos_score = self.model.score(seq_t, pos_t)
                neg_score = self.model.score(seq_t, neg_t)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if losses:
                print(f"SASRec epoch {epoch:02d}/{self.epochs}: loss={float(np.mean(losses)):.4f}")

        self.model.eval()
        self.model.to("cpu")
        self.device = "cpu"
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        if not hasattr(self, "model") or str(user_id) not in self.histories:
            return []
        all_indices = sorted(self.idx_to_item)
        with torch.no_grad():
            seq_t = torch.tensor([_pad_left(self.histories.get(str(user_id), []), self.max_seq_len)], dtype=torch.long)
            item_t = torch.tensor(all_indices, dtype=torch.long)
            seq_vec = self.model.sequence_vector(seq_t)
            scores_np = (self.model.item_emb(item_t) @ seq_vec.reshape(-1)).detach().numpy()
        order = np.argsort(-scores_np)[: min(len(scores_np), top_k * 5)]
        scores = [(self.idx_to_item[all_indices[idx]], float(scores_np[idx])) for idx in order]
        return self._filter_seen(str(user_id), scores, top_k)

    def init_kwargs(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "max_seq_len": self.max_seq_len,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    def _save_state(self, target: Path) -> None:
        torch.save(self.model.state_dict(), target / "state.pt")
        with (target / "state.pkl").open("wb") as file:
            pickle.dump(
                {
                    "user_to_idx": self.user_to_idx,
                    "item_to_idx": self.item_to_idx,
                    "idx_to_item": self.idx_to_item,
                    "histories": self.histories,
                    "embedding_dim": self.embedding_dim,
                    "max_seq_len": self.max_seq_len,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
                file,
            )

    def _load_state(self, target: Path) -> None:
        with (target / "state.pkl").open("rb") as file:
            state = pickle.load(file)
        self.user_to_idx = {str(k): int(v) for k, v in state["user_to_idx"].items()}
        self.item_to_idx = {str(k): int(v) for k, v in state["item_to_idx"].items()}
        self.idx_to_item = {int(k): str(v) for k, v in state["idx_to_item"].items()}
        self.histories = {
            str(user_id): [int(item_idx) for item_idx in item_indices]
            for user_id, item_indices in state.get("histories", {}).items()
        }
        self.embedding_dim = int(state.get("embedding_dim", self.embedding_dim))
        self.max_seq_len = int(state.get("max_seq_len", self.max_seq_len))
        self.num_heads = int(state.get("num_heads", self.num_heads))
        self.num_layers = int(state.get("num_layers", self.num_layers))
        self.dropout = float(state.get("dropout", self.dropout))
        self.model = _SASRec(
            max(1, len(self.idx_to_item)),
            self.embedding_dim,
            self.max_seq_len,
            self.num_heads,
            self.num_layers,
            self.dropout,
        )
        self.model.load_state_dict(torch.load(target / "state.pt", map_location="cpu"))
        self.model.eval()


def _pad_left(values: list[int], length: int) -> list[int]:
    values = values[-length:]
    return [0] * (length - len(values)) + values
