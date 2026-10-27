from llm4rec.retrieval.base import BaseRetriever, Candidate
from llm4rec.retrieval.bpr import BPRMFRetriever
from llm4rec.retrieval.content_faiss import ContentFaissRetriever
from llm4rec.retrieval.exposure import ExposureQualityRetriever
from llm4rec.retrieval.itemcf import ItemCFRetriever
from llm4rec.retrieval.lightgcn import LightGCNRetriever
from llm4rec.retrieval.merge import reciprocal_rank_fusion, weighted_merge
from llm4rec.retrieval.metadata import MetadataRetriever
from llm4rec.retrieval.popular import PopularRetriever
from llm4rec.retrieval.sasrec import SASRecRetriever
from llm4rec.retrieval.two_tower import TwoTowerRetriever
from llm4rec.retrieval.trend import TrendRetriever

RETRIEVER_REGISTRY = {
    "popular": PopularRetriever,
    "content_faiss": ContentFaissRetriever,
    "exposure": ExposureQualityRetriever,
    "itemcf": ItemCFRetriever,
    "bpr": BPRMFRetriever,
    "two_tower": TwoTowerRetriever,
    "lightgcn": LightGCNRetriever,
    "sasrec": SASRecRetriever,
    "metadata": MetadataRetriever,
    "trend": TrendRetriever,
}

__all__ = [
    "BaseRetriever",
    "Candidate",
    "PopularRetriever",
    "ContentFaissRetriever",
    "ExposureQualityRetriever",
    "ItemCFRetriever",
    "LightGCNRetriever",
    "MetadataRetriever",
    "TrendRetriever",
    "BPRMFRetriever",
    "TwoTowerRetriever",
    "SASRecRetriever",
    "weighted_merge",
    "reciprocal_rank_fusion",
    "RETRIEVER_REGISTRY",
]
