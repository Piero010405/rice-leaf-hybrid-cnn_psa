from __future__ import annotations

import argparse
from riceleaf.config.loader import load_config
from riceleaf.data.make_splits import make_splits, split_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    split = cfg.data.split
    make_splits(
        raw_dir=cfg.data.raw_dir,
        split_dir=cfg.data.split_dir,
        class_names=list(cfg.data.class_names),
        extensions=list(cfg.data.extensions),
        train_ratio=float(split.train),
        val_ratio=float(split.val),
        test_ratio=float(split.test),
        seed=int(cfg.project.seed),
    )
    print(split_summary(cfg.data.split_dir))


if __name__ == "__main__":
    main()
