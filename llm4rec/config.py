from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    video_csv: Path = PROJECT_ROOT / "data" / "mini_youtube8m.csv"
    user_behavior_json: Path = PROJECT_ROOT / "data" / "user_behavior.json"
    ctr_samples_csv: Path = PROJECT_ROOT / "data" / "train_ctr_samples.csv"
    video_embeddings_npy: Path = PROJECT_ROOT / "data" / "video_embeddings.npy"
    faiss_index: Path = PROJECT_ROOT / "data" / "video_embeddings.faiss"
    video_id_map: Path = PROJECT_ROOT / "data" / "video_id_map.txt"
    ctr_model: Path = PROJECT_ROOT / "models" / "ctr_mlp_model.pt"
    ctr_metrics: Path = PROJECT_ROOT / "models" / "ctr_metrics.json"

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root_path = Path(root)
        return cls(
            root=root_path,
            data_dir=root_path / "data",
            models_dir=root_path / "models",
            video_csv=root_path / "data" / "mini_youtube8m.csv",
            user_behavior_json=root_path / "data" / "user_behavior.json",
            ctr_samples_csv=root_path / "data" / "train_ctr_samples.csv",
            video_embeddings_npy=root_path / "data" / "video_embeddings.npy",
            faiss_index=root_path / "data" / "video_embeddings.faiss",
            video_id_map=root_path / "data" / "video_id_map.txt",
            ctr_model=root_path / "models" / "ctr_mlp_model.pt",
            ctr_metrics=root_path / "models" / "ctr_metrics.json",
        )


DEFAULT_PATHS = ProjectPaths()


def ensure_project_dirs(paths: ProjectPaths = DEFAULT_PATHS) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
