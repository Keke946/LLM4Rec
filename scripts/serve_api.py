from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
    raise SystemExit(
        "FastAPI service dependencies are missing. Install requirements.txt or run `pip install fastapi uvicorn`."
    ) from exc

from llm4rec.recommender import LLM4RecRecommender


class RecommendRequest(BaseModel):
    user_id: str | None = None
    query: str | None = None
    user_interest: str | list[str] | None = None
    top_k: int = 10
    candidate_pool_k: int = 500
    retrievers: list[str] | None = None
    ranker: str = "deepfm"
    rerank_mode: str = "retrieval_ranker_fusion"
    retrieval_weight: float = 0.7
    ranker_weight: float = 0.3
    reranker: str = "none"
    debug: bool = False


DATASET = os.getenv("LLM4REC_DATASET", "movielens")
PROCESSED_DIR = os.getenv("LLM4REC_PROCESSED_DIR", f"data/processed/{DATASET}")
RETRIEVER_DIR = os.getenv("LLM4REC_RETRIEVER_DIR", f"models/{DATASET}/retrievers")

app = FastAPI(title="LLM4Rec API", version="0.1.0")
recommender = LLM4RecRecommender(
    dataset=DATASET,
    processed_dir=PROCESSED_DIR,
    retriever_dir=RETRIEVER_DIR,
) if Path(PROCESSED_DIR).exists() else LLM4RecRecommender()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dataset": DATASET if Path(PROCESSED_DIR).exists() else "synthetic_fallback",
        "processed_dir": PROCESSED_DIR,
    }


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    results = recommender.recommend(
        user_interest=request.user_interest,
        user_id=request.user_id,
        query=request.query,
        top_k=request.top_k,
        candidate_pool_k=request.candidate_pool_k,
        retrievers=request.retrievers,
        ranker=request.ranker,
        rerank_mode=request.rerank_mode,
        retrieval_weight=request.retrieval_weight,
        ranker_weight=request.ranker_weight,
        reranker=request.reranker,
        debug=request.debug,
    )
    return {"results": results}


@app.get("/debug/user/{user_id}")
def debug_user(user_id: str) -> dict[str, Any]:
    if recommender.processed_interactions is None:
        return {"user_id": user_id, "mode": "synthetic_fallback", "message": "No processed dataset loaded."}
    user_rows = recommender.processed_interactions[
        recommender.processed_interactions["user_id"].astype(str) == str(user_id)
    ].sort_values(["timestamp", "item_id"])
    positives = user_rows[user_rows["label"] == 1]
    split_counts = user_rows["split"].value_counts().to_dict()
    seen_train_val = positives[positives["split"].isin(["train", "val"])]["item_id"].astype(str).tolist()
    sample_recommendations = recommender.recommend(
        user_id=user_id,
        top_k=5,
        candidate_pool_k=100,
        retrievers=["popular", "itemcf"],
        ranker="deepfm",
        debug=True,
    )
    return {
        "user_id": user_id,
        "num_interactions": int(len(user_rows)),
        "num_positive_interactions": int(len(positives)),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "seen_train_val_positive_items": seen_train_val[:50],
        "recent_positive_items": positives["item_id"].astype(str).tail(20).tolist(),
        "sample_recommendations": sample_recommendations,
    }


if __name__ == "__main__":
    host = os.getenv("LLM4REC_HOST", "0.0.0.0")
    port = int(os.getenv("LLM4REC_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
