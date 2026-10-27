from llm4rec.datasets.base import BaseDatasetAdapter, DatasetMeta
from llm4rec.datasets.kuairec import KuaiRecAdapter
from llm4rec.datasets.mind import MINDAdapter
from llm4rec.datasets.movielens import MovieLensAdapter

DATASET_ADAPTERS = {
    "kuairec": KuaiRecAdapter,
    "mind": MINDAdapter,
    "movielens": MovieLensAdapter,
}

__all__ = [
    "BaseDatasetAdapter",
    "DatasetMeta",
    "KuaiRecAdapter",
    "MINDAdapter",
    "MovieLensAdapter",
    "DATASET_ADAPTERS",
]
