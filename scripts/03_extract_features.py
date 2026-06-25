from __future__ import annotations

import argparse
from pathlib import Path
from riceleaf.config.loader import load_config
from riceleaf.features.extractor import extract_features_from_split
from riceleaf.features.scaler import fit_transform_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = cfg.features.get("source", cfg.data.get("texture_source", "original"))
    image_root = Path(cfg.data.segmented_dir) if source == "segmented" else Path(cfg.data.raw_dir)
    out_dir = Path(cfg.data.features_dir) / f"glcm_dwt_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ["train", "val", "test"]:
        out_csv = out_dir / f"{split}_features.csv"
        extract_features_from_split(Path(cfg.data.split_dir) / f"{split}.csv", image_root, cfg, out_csv)
        paths[split] = out_csv
    fit_transform_save(paths["train"], paths["val"], paths["test"], out_dir)
    print(f"Features saved at {out_dir}")


if __name__ == "__main__":
    main()
