from llm4rec.ranking.ctr_mlp import CTRMLPRanker
from llm4rec.ranking.deepfm import DeepFM
from llm4rec.ranking.din import DIN
from llm4rec.ranking.embedding_mlp import EmbeddingMLP

RANKER_REGISTRY = {
    "ctr_mlp": CTRMLPRanker,
    "embedding_mlp": EmbeddingMLP,
    "deepfm": DeepFM,
    "din": DIN,
}

__all__ = ["CTRMLPRanker", "EmbeddingMLP", "DeepFM", "DIN", "RANKER_REGISTRY"]
