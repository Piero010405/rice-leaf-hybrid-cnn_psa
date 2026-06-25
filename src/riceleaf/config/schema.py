from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    name: str = "rice-leaf-hybrid-cnn"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs"


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/rice_leaf_disease"
    processed_dir: str = "data/processed"
    segmented_dir: str = "data/segmented"
    features_dir: str = "data/features"
    split_dir: str = "data/splits"
    image_size: int = 224
    num_classes: int = 4
    class_names: list[str] = field(default_factory=list)
