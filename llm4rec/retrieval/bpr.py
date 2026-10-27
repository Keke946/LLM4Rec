from __future__ import annotations

import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm4rec.retrieval.base import BaseRetriever, Candidate

from llm4rec.retrieval.sampling import sample_negative


class _BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)


class BPRDataset(Dataset):
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


class BPRMFRetriever(BaseRetriever):
    name = "bpr"

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        lr: float = 1e-2,
        seed: int = 42,
        batch_size: int = 1024,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "BPRMFRetriever":
        super().fit(train_interactions, items)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        users = sorted(train_interactions["user_id"].astype(str).unique())
        item_ids = sorted(items["item_id"].astype(str).unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_item = item_ids

        positives = train_interactions[train_interactions["label"] == 1]
        positive_pairs = []
        all_item_indices = set(range(len(item_ids)))
        positives_by_user_idx: dict[int, set[int]] = {}
        for user_id, group in positives.groupby("user_id"):
            user_idx = self.user_to_idx[str(user_id)]
            pos_items = {self.item_to_idx[str(item_id)] for item_id in group["item_id"].astype(str) if str(item_id) in self.item_to_idx}
            positives_by_user_idx[user_idx] = pos_items
            for pos_idx in pos_items:
                if len(pos_items) < len(all_item_indices):
                    positive_pairs.append((user_idx, pos_idx))

        self.model = _BPRMF(max(1, len(users)), max(1, len(item_ids)), self.embedding_dim).to(self.device)
        if not positive_pairs:
            return self

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = BPRDataset(positive_pairs, positives_by_user_idx, len(item_ids), seed=self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            for users_t, pos_t, neg_t in loader:
                users_t = users_t.to(self.device)
                pos_t = pos_t.to(self.device)
                neg_t = neg_t.to(self.device)
                pos_score = self.model.score(users_t, pos_t)
                neg_score = self.model.score(users_t, neg_t)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if losses:
                print(f"BPR epoch {epoch:02d}/{self.epochs}: loss={float(np.mean(losses)):.4f}")
        self.model.eval()
        self.model.to("cpu")
        self.device = "cpu"
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        if not hasattr(self, "model") or str(user_id) not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[str(user_id)]
        with torch.no_grad():
            users = torch.full((len(self.idx_to_item),), user_idx, dtype=torch.long)
            items = torch.arange(len(self.idx_to_item), dtype=torch.long)
            scores_np = self.model.score(users, items).detach().numpy()
        order = np.argsort(-scores_np)[: min(len(scores_np), top_k * 5)]
        scores = [(self.idx_to_item[idx], float(scores_np[idx])) for idx in order]
        return self._filter_seen(str(user_id), scores, top_k)

    def init_kwargs(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "lr": self.lr,
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
        self.model = _BPRMF(
            max(1, len(self.user_to_idx)),
            max(1, len(self.idx_to_item)),
            self.embedding_dim,
        )
        self.model.load_state_dict(torch.load(target / "state.pt", map_location="cpu"))
        self.model.eval()
