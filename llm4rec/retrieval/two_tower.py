from __future__ import annotations

import random
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from llm4rec.retrieval.base import BaseRetriever, Candidate

from llm4rec.retrieval.sampling import sample_negative


class _TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.user_proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def user_vector(self, users: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(users)
        hist_emb = self.item_emb(histories)
        mask = (histories != 0).float().unsqueeze(-1)
        hist_vec = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.user_proj(torch.cat([user_vec, hist_vec], dim=-1)), dim=-1)

    def item_vector(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_emb(items), dim=-1)


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        triples: list[tuple[int, str, int]],
        positives_by_user: dict[str, set[int]],
        item_indices: list[int],
        histories: dict[str, list[int]],
        max_history_len: int,
        seed: int = 42,
    ) -> None:
        self.triples = triples
        self.positives_by_user = positives_by_user
        self.item_indices = item_indices
        self.histories = histories
        self.max_history_len = max_history_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int):
        user_idx, user_id, pos_idx = self.triples[index]
        positives = self.positives_by_user.get(user_id, set())
        neg_idx = sample_negative(self.rng, self.item_indices, positives)
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(_pad(self.histories.get(user_id, []), self.max_history_len), dtype=torch.long),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


class TwoTowerRetriever(BaseRetriever):
    name = "two_tower"

    def __init__(
        self,
        embedding_dim: int = 32,
        epochs: int = 5,
        lr: float = 1e-2,
        max_history_len: int = 20,
        seed: int = 42,
        batch_size: int = 1024,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr
        self.max_history_len = max_history_len
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_interactions: pd.DataFrame, items: pd.DataFrame) -> "TwoTowerRetriever":
        super().fit(train_interactions, items)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        users = sorted(train_interactions["user_id"].astype(str).unique())
        item_ids = sorted(items["item_id"].astype(str).unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}
        self.idx_to_item = {idx + 1: item_id for idx, item_id in enumerate(item_ids)}

        positives = train_interactions[train_interactions["label"] == 1]
        self.histories: dict[str, list[int]] = {}
        for user_id, group in positives.groupby("user_id"):
            self.histories[str(user_id)] = [
                self.item_to_idx[str(item_id)]
                for item_id in group.sort_values("timestamp")["item_id"].astype(str)
                if str(item_id) in self.item_to_idx
            ][-self.max_history_len :]

        all_item_indices = set(self.idx_to_item)
        triples = []
        positives_by_user_idx: dict[str, set[int]] = {}
        for user_id, pos_items_raw in self.train_positive_by_user.items():
            if user_id not in self.user_to_idx:
                continue
            pos_items = {self.item_to_idx[item_id] for item_id in pos_items_raw if item_id in self.item_to_idx}
            positives_by_user_idx[user_id] = pos_items
            for pos_idx in pos_items:
                if len(pos_items) < len(all_item_indices):
                    triples.append((self.user_to_idx[user_id], user_id, pos_idx))

        self.model = _TwoTower(max(1, len(users)), max(1, len(item_ids) + 1), self.embedding_dim).to(self.device)
        if not triples:
            return self

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = TwoTowerDataset(
            triples,
            positives_by_user_idx,
            sorted(all_item_indices),
            self.histories,
            self.max_history_len,
            seed=self.seed,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            for users_t, hist_t, pos_t, neg_t in loader:
                users_t = users_t.to(self.device)
                hist_t = hist_t.to(self.device)
                pos_t = pos_t.to(self.device)
                neg_t = neg_t.to(self.device)
                user_vec = self.model.user_vector(users_t, hist_t)
                pos_vec = self.model.item_vector(pos_t)
                neg_vec = self.model.item_vector(neg_t)
                pos_score = (user_vec * pos_vec).sum(dim=-1)
                neg_score = (user_vec * neg_vec).sum(dim=-1)
                loss = -F.logsigmoid(pos_score - neg_score).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            if losses:
                print(f"TwoTower epoch {epoch:02d}/{self.epochs}: loss={float(np.mean(losses)):.4f}")
        self.model.eval()
        self.model.to("cpu")
        self.device = "cpu"
        return self

    def retrieve(self, user_id: str, query: str | None = None, top_k: int = 100) -> list[Candidate]:
        if not hasattr(self, "model") or str(user_id) not in self.user_to_idx:
            return []
        user_idx = self.user_to_idx[str(user_id)]
        all_indices = sorted(self.idx_to_item)
        with torch.no_grad():
            hist_t = torch.tensor([_pad(self.histories.get(str(user_id), []), self.max_history_len)], dtype=torch.long)
            user_vec = self.model.user_vector(torch.tensor([user_idx], dtype=torch.long), hist_t)
            item_vecs = self.model.item_vector(torch.tensor(all_indices, dtype=torch.long))
            scores_np = (item_vecs @ user_vec.reshape(-1)).detach().numpy()
        order = np.argsort(-scores_np)[: min(len(scores_np), top_k * 5)]
        scores = [(self.idx_to_item[all_indices[idx]], float(scores_np[idx])) for idx in order]
        return self._filter_seen(str(user_id), scores, top_k)

    def init_kwargs(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "max_history_len": self.max_history_len,
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
                    "max_history_len": self.max_history_len,
                },
                file,
            )
        self._export_item_embeddings(target)

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
        self.max_history_len = int(state.get("max_history_len", self.max_history_len))
        self.model = _TwoTower(
            max(1, len(self.user_to_idx)),
            max(1, len(self.idx_to_item) + 1),
            self.embedding_dim,
        )
        self.model.load_state_dict(torch.load(target / "state.pt", map_location="cpu"))
        self.model.eval()

    def _export_item_embeddings(self, target: Path) -> None:
        if not hasattr(self, "model") or not getattr(self, "idx_to_item", None):
            return
        all_indices = sorted(self.idx_to_item)
        with torch.no_grad():
            embeddings = self.model.item_vector(torch.tensor(all_indices, dtype=torch.long)).detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(target / "item_embeddings.npy", embeddings)
        (target / "item_embedding_ids.json").write_text(
            json.dumps([self.idx_to_item[idx] for idx in all_indices], ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(target / "item_embeddings.faiss"))
        except Exception:
            return


def _pad(values: list[int], length: int) -> list[int]:
    values = values[-length:]
    return values + [0] * (length - len(values))
