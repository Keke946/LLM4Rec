from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from llm4rec.config import DEFAULT_PATHS, ensure_project_dirs
from llm4rec.data import load_videos
from llm4rec.retrieval.content_faiss import ContentFaissRetriever


def build_retrieval_index(model_name: str, batch_size: int) -> None:
    ensure_project_dirs()
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Vector index dependencies are missing. Install them with "
            "`pip install faiss-cpu sentence-transformers`."
        ) from exc

    videos = load_videos()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        videos["full_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(DEFAULT_PATHS.video_embeddings_npy, embeddings)
    faiss.write_index(index, str(DEFAULT_PATHS.faiss_index))
    DEFAULT_PATHS.video_id_map.write_text(
        "\n".join(videos["video_id"].tolist()) + "\n",
        encoding="utf-8",
    )
    print(f"Saved embeddings to {DEFAULT_PATHS.video_embeddings_npy}")
    print(f"Saved FAISS index to {DEFAULT_PATHS.faiss_index}")
    print(f"Saved id map to {DEFAULT_PATHS.video_id_map}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SentenceTransformer + FAISS recall index.")
    parser.add_argument("--dataset", default=None, help="Dataset name for processed-data retrieval index.")
    parser.add_argument("--model", default="content_faiss", help="Retriever model name for dataset mode.")
    parser.add_argument("--processed_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--output_dir", default=None, help="Retriever output directory.")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="SentenceTransformer model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    args = parser.parse_args()
    if args.dataset:
        processed_dir = Path(args.processed_dir or f"data/processed/{args.dataset}")
        output_dir = Path(args.output_dir or f"models/{args.dataset}/retrievers")
        output_dir.mkdir(parents=True, exist_ok=True)
        items = pd.read_parquet(processed_dir / "items.parquet")
        interactions = pd.read_parquet(processed_dir / "interactions.parquet")
        train = interactions[interactions["split"] == "train"].reset_index(drop=True)
        retriever = ContentFaissRetriever(model_name=args.model_name).fit(train, items)
        path = output_dir / f"{args.model}.pkl"
        retriever.save(path)
        print(f"Saved {args.model} retrieval index to {path}")
        return
    try:
        build_retrieval_index(args.model_name, args.batch_size)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
