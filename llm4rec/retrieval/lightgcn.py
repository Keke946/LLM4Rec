from __future__ import annotations

import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm4rec.retrieval.base import BaseRetriever, Candidate

from llm4rec.retrieval.sampling import sample_negative


class _PairwiseDataset(Dataset):
    def __init__(
        self,
        positive_pairs: list[tuple[int, int]],
        positives_by_user: dict[int, set[int]],
        num_items: int,
        seed: int = 42,
    ) -> None:
        self.positive_pairs = positive_pairs
        self.positives_by_user = positives_by_user
        self.num_items = num_items
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, pos_idx = self.positive_pairs[index]
        positives = self.positives_by_user.get(user_idx, set())
        neg_idx = sample_negative(self.rng, range(self.num_items), positives)
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


class _LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        norm_adj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.register_buffer("norm_adj", norm_adj.coalesce())
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layer_outputs = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            layer_outputs.append(embeddings)
        final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vecs, item_vecs = self.all_embeddings()
        return (user_vecs[users] * item_vecs[items]).sum(dim=-1)


class LightGCNRetriever(BaseRetriever):
    name = "lightgcn"

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        lr: float = 1e-2,
        num_layers: int = 2,
        reg_weight: float = 1e-4,
        seed: int = 42,
        batch_size: int = 1024,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.num_layers = num_layers
        self.reg_weight = reg_weight
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "LightGCNRetriever":
        super().fit(train_interactions, items)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        users = sorted(train_interactions["user_id"].astype(str).unique())
        item_ids = sorted(items["item_id"].astype(str).unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_item = item_ids

        positives = train_interactions[train_interactions["label"] == 1]
        positive_pairs: list[tuple[int, int]] = []
        positives_by_user_idx: dict[int, set[int]] = {}
        for user_id, group in positives.groupby("user_id"):
            user_idx = self.user_to_idx[str(user_id)]
            pos_items = {
                self.item_to_idx[str(item_id)]
                for item_id in group["item_id"].astype(str)
                if str(item_id) in self.item_to_idx
            }
            positives_by_user_idx[user_idx] = pos_items
            for item_idx in pos_items:
                if len(pos_items) < len(item_ids):
                    positive_pairs.append((user_idx, item_idx))

        self.norm_adj = _build_norm_adj(max(1, len(users)), max(1, len(item_ids)), positive_pairs)
        self.model = _LightGCN(
            max(1, len(users)),
            max(1, len(item_ids)),
            self.embedding_dim,
            self.num_layers,
            self.norm_adj,
        ).to(self.device)
        if not positive_pairs:
            self.model.to("cpu")
            self.device = "cpu"
            return self

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = _PairwiseDataset(positive_pairs, positives_by_user_idx, len(item_ids), seed=self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            for users_t, pos_t, neg_t in loader:
                users_t = users_t.to(self.device)
                pos_t = pos_t.to(self.device)
                neg_t = neg_t.to(self.device)
                pos_score = self.model.score(users_t, pos_t)
                neg_score = self.model.score(users_t, neg_t)
                bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
                reg_loss = (
                    self.model.user_emb(users_t).norm(2).pow(2)
                    + self.model.item_emb(pos_t).norm(2).pow(2)
                    + self.model.item_emb(neg_t).norm(2).pow(2)
                ) / max(1, len(users_t))
                loss = bpr_loss + self.reg_weight * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if losses:
                print(f"LightGCN epoch {epoch:02d}/{self.epochs}: loss={float(np.mean(losses)):.4f}")

        self.model.eval()
        self.model.to("cpu")
        self.device = "cpu"
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        if not hasattr(self, "model") or str(user_id) not in self.user_to_idx or not self.idx_to_item:
            return []
        user_idx = self.user_to_idx[str(user_id)]
        with torch.no_grad():
            user_vecs, item_vecs = self.model.all_embeddings()
            scores_np = (item_vecs @ user_vecs[user_idx]).detach().cpu().numpy()
        order = np.argsort(-scores_np)[: min(len(scores_np), top_k * 5)]
        scores = [(self.idx_to_item[idx], float(scores_np[idx])) for idx in order]
        return self._filter_seen(str(user_id), scores, top_k)

    def init_kwargs(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "num_layers": self.num_layers,
            "reg_weight": self.reg_weight,
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
                    "embedding_dim": self.embedding_dim,
                    "num_layers": self.num_layers,
                    "reg_weight": self.reg_weight,
                    "positive_pairs": self._positive_pairs_for_graph(),
                },
                file,
            )

    def _load_state(self, target: Path) -> None:
        with (target / "state.pkl").open("rb") as file:
            state = pickle.load(file)
        self.user_to_idx = {str(k): int(v) for k, v in state["user_to_idx"].items()}
        self.item_to_idx = {str(k): int(v) for k, v in state["item_to_idx"].items()}
        self.idx_to_item = [str(item_id) for item_id in state["idx_to_item"]]
        self.embedding_dim = int(state.get("embedding_dim", self.embedding_dim))
        self.num_layers = int(state.get("num_layers", self.num_layers))
        self.reg_weight = float(state.get("reg_weight", self.reg_weight))
        positive_pairs = [(int(user), int(item)) for user, item in state.get("positive_pairs", [])]
        self.norm_adj = _build_norm_adj(
            max(1, len(self.user_to_idx)),
            max(1, len(self.idx_to_item)),
            positive_pairs,
        )
        self.model = _LightGCN(
            max(1, len(self.user_to_idx)),
            max(1, len(self.idx_to_item)),
            self.embedding_dim,
            self.num_layers,
            self.norm_adj,
        )
        self.model.load_state_dict(torch.load(target / "state.pt", map_location="cpu"))
        self.model.eval()

    def _positive_pairs_for_graph(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for user_id, item_ids in self.train_positive_by_user.items():
            if user_id not in self.user_to_idx:
                continue
            user_idx = self.user_to_idx[user_id]
            for item_id in item_ids:
                if item_id in self.item_to_idx:
                    pairs.append((user_idx, self.item_to_idx[item_id]))
        return pairs


def _build_norm_adj(num_users: int, num_items: int, positive_pairs: list[tuple[int, int]]) -> torch.Tensor:
    size = num_users + num_items
    if not positive_pairs:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
        return _sparse_tensor(indices, values, (size, size))

    rows: list[int] = []
    cols: list[int] = []
    degrees = np.zeros(size, dtype=np.float32)
    for user_idx, item_idx in positive_pairs:
        item_node = num_users + item_idx
        rows.extend([user_idx, item_node])
        cols.extend([item_node, user_idx])
        degrees[user_idx] += 1.0
        degrees[item_node] += 1.0

    values: list[float] = []
    for row, col in zip(rows, cols):
        denom = float(np.sqrt(max(degrees[row], 1.0) * max(degrees[col], 1.0)))
        values.append(1.0 / denom)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    return _sparse_tensor(indices, value_tensor, (size, size))


def _sparse_tensor(indices: torch.Tensor, values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled.*")
        try:
            return torch.sparse_coo_tensor(
                indices,
                values,
                size,
                check_invariants=False,
            ).coalesce()
        except TypeError:
            return torch.sparse_coo_tensor(indices, values, size).coalesce()
