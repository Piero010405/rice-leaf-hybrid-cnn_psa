from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_image_transform

META_COLS = {"image_id", "relative_path", "label", "label_idx"}


class RiceLeafDataset(Dataset):
    """Dataset for RGB-only or hybrid image+feature training."""

    def __init__(self, split_csv: str | Path, image_root: str | Path, image_size: int, train: bool = False, features_csv: str | Path | None = None):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = build_image_transform(image_size, train=train)
        self.features = None
        self.feature_cols: list[str] = []
        if features_csv:
            fdf = pd.read_csv(features_csv)
            self.feature_cols = [c for c in fdf.columns if c not in META_COLS]
            self.features = fdf.set_index("relative_path")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        rel = row["relative_path"]
        path = self.image_root / rel
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        label = int(row["label_idx"])
        item: dict[str, Any] = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "relative_path": rel,
        }
        if self.features is not None:
            feat_row = self.features.loc[rel]
            values = feat_row[self.feature_cols].astype("float32").values
            item["features"] = torch.tensor(values, dtype=torch.float32)
        return item
