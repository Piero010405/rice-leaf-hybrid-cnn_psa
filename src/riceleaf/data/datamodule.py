from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from .dataset import RiceLeafDataset


def build_dataloaders(cfg: DictConfig):
    split_dir = Path(cfg.data.split_dir)
    if cfg.data.get("use_segmented_images", False):
        image_root = Path(cfg.data.segmented_dir)
    else:
        image_root = Path(cfg.data.raw_dir)

    source = cfg.data.get("texture_source", cfg.features.get("source", "original"))
    features_dir = Path(cfg.data.features_dir) / f"glcm_dwt_{source}"
    use_features = cfg.data.get("use_texture_features", False)
    def feat_csv(name: str):
        p = features_dir / f"{name}_features_scaled.csv"
        return p if use_features and p.exists() else None

    train_ds = RiceLeafDataset(split_dir/"train.csv", image_root, cfg.data.image_size, train=True, features_csv=feat_csv("train"))
    val_ds = RiceLeafDataset(split_dir/"val.csv", image_root, cfg.data.image_size, train=False, features_csv=feat_csv("val"))
    test_ds = RiceLeafDataset(split_dir/"test.csv", image_root, cfg.data.image_size, train=False, features_csv=feat_csv("test"))

    kwargs = dict(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, shuffle=True, **kwargs),
        "val": DataLoader(val_ds, shuffle=False, **kwargs),
        "test": DataLoader(test_ds, shuffle=False, **kwargs),
        "feature_dim": len(train_ds.feature_cols),
    }
