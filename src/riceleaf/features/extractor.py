from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from riceleaf.preprocessing.image_io import read_rgb
from riceleaf.preprocessing.resize import resize_image
from .glcm import extract_glcm_features
from .dwt import extract_dwt_features


def extract_features_for_image(path: str | Path, cfg: DictConfig) -> dict[str, float]:
    rgb = read_rgb(path)
    rgb = resize_image(rgb, int(cfg.data.image_size))
    methods = list(cfg.features.get("methods", []))
    features: dict[str, float] = {}
    if "glcm" in methods:
        glcm_cfg = cfg.features.get("glcm", {})
        features.update(extract_glcm_features(rgb, **dict(glcm_cfg)))
    if "dwt" in methods:
        dwt_cfg = cfg.features.get("dwt", {})
        features.update(extract_dwt_features(rgb, **dict(dwt_cfg)))
    return features


def extract_features_from_split(split_csv: str | Path, image_root: str | Path, cfg: DictConfig, out_csv: str | Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    rows = []
    image_root = Path(image_root)
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"features:{Path(out_csv).stem}"):
        rel = row["relative_path"]
        img_path = image_root / rel
        feats = extract_features_for_image(img_path, cfg)
        feats.update({"image_id": row.get("image_id", Path(rel).stem), "relative_path": rel, "label": row["label"], "label_idx": row["label_idx"]})
        rows.append(feats)
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
