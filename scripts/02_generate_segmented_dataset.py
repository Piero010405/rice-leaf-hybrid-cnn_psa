from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from riceleaf.config.loader import load_config
from riceleaf.preprocessing.image_io import read_rgb, save_rgb
from riceleaf.preprocessing.resize import resize_image
from riceleaf.segmentation.lab_otsu import segment_lab_otsu
from riceleaf.segmentation.visualize import save_segmentation_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_root = Path(cfg.data.raw_dir)
    out_root = Path(cfg.data.segmented_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    example_count = 0
    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(cfg.data.split_dir) / f"{split}.csv")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"segment:{split}"):
            src = raw_root / row["relative_path"]
            dst = out_root / row["relative_path"]
            rgb = read_rgb(src)
            rgb = resize_image(rgb, int(cfg.data.image_size))
            seg_cfg = cfg.segmentation
            morph = seg_cfg.get("morphology", {})
            segmented, mask = segment_lab_otsu(
                rgb,
                channel=seg_cfg.get("channel", "a"),
                invert_mask=seg_cfg.get("invert_mask", False),
                kernel_size=morph.get("kernel_size", 5),
                opening=morph.get("opening", True),
                closing=morph.get("closing", True),
                iterations=morph.get("iterations", 1),
            )
            save_rgb(dst, segmented)
            if example_count < args.max_examples:
                out_panel = Path("outputs/figures/segmentation_examples") / f"{split}_{example_count:02d}.png"
                save_segmentation_panel(rgb, mask, segmented, out_panel)
                example_count += 1
    print(f"Segmented images saved at {out_root}")


if __name__ == "__main__":
    main()
