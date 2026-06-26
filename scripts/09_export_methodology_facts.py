from pathlib import Path
import json
import pandas as pd

SPLIT_DIR = Path("data/splits")
FEATURE_DIRS = [
    Path("data/features/glcm_dwt_original"),
    Path("data/features/glcm_dwt_segmented"),
]
OUT = Path("outputs/reports/methodology_facts.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)


def write_line(f, text=""):
    print(text)
    f.write(str(text) + "\n")


def load_split(split):
    path = SPLIT_DIR / f"{split}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    with OUT.open("w", encoding="utf-8") as f:
        write_line(f, "=== METHODOLOGY FACTS ===")
        write_line(f)

        write_line(f, "=== SPLIT DISTRIBUTION ===")
        total = 0
        all_rows = []

        for split in ["train", "val", "test"]:
            df = load_split(split)
            if df is None:
                write_line(f, f"{split}: NOT FOUND")
                continue

            total += len(df)
            write_line(f, f"{split}: {len(df)} images")
            write_line(f, df["label"].value_counts().sort_index().to_string())
            write_line(f)

            temp = df.copy()
            temp["split"] = split
            all_rows.append(temp)

        write_line(f, f"Total images across splits: {total}")
        write_line(f)

        if all_rows:
            all_df = pd.concat(all_rows, ignore_index=True)

            write_line(f, "=== COLUMNS IN SPLITS ===")
            write_line(f, all_df.columns.tolist())
            write_line(f)

            if "phash" in all_df.columns:
                write_line(f, "=== PHASH VALIDATION ===")
                write_line(f, f"Total images: {len(all_df)}")
                write_line(f, f"Unique pHashes: {all_df['phash'].nunique()}")

                cross_split = (
                    all_df.groupby("phash")["split"]
                    .nunique()
                    .reset_index(name="n_splits")
                )
                overlap = cross_split[cross_split["n_splits"] > 1]

                write_line(f, f"pHashes appearing in more than one split: {len(overlap)}")
                write_line(f)

        write_line(f, "=== FEATURE FILES ===")
        for feature_dir in FEATURE_DIRS:
            write_line(f, f"Directory: {feature_dir}")

            if not feature_dir.exists():
                write_line(f, "NOT FOUND")
                write_line(f)
                continue

            for split in ["train", "val", "test"]:
                candidates = [
                    feature_dir / f"{split}_features_scaled.csv",
                    feature_dir / f"{split}_features.csv",
                ]

                found = False
                for path in candidates:
                    if path.exists():
                        df = pd.read_csv(path)
                        write_line(f, f"{path}: rows={len(df)}, columns={len(df.columns)}")
                        write_line(f, f"Columns: {df.columns.tolist()[:20]}{' ...' if len(df.columns) > 20 else ''}")
                        found = True
                        break

                if not found:
                    write_line(f, f"{split}: feature file not found")

            write_line(f)

        write_line(f, "=== RUNS AND METRICS FILES ===")
        metric_files = sorted(Path("runs").glob("**/metrics.json"))

        write_line(f, f"metrics.json files found: {len(metric_files)}")

        for mf in metric_files[:5]:
            write_line(f, str(mf))

        if len(metric_files) > 5:
            write_line(f, "...")

        write_line(f)

        write_line(f, "=== CONFIG SNAPSHOT ===")
        config_files = [
            Path("configs/default.yml"),
            Path("configs/experiments/E0_baseline_cnn.yml"),
            Path("configs/experiments/E1_segmented_cnn.yml"),
            Path("configs/experiments/E2_cnn_texture.yml"),
            Path("configs/experiments/E3_proposed_segmented_cnn_texture.yml"),
        ]

        for path in config_files:
            write_line(f, f"--- {path} ---")

            if not path.exists():
                write_line(f, "NOT FOUND")
                continue

            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                lower = line.lower()
                keys = [
                    "seed", "batch", "epoch", "learning", "lr",
                    "optimizer", "scheduler", "image", "size",
                    "mobilenet", "texture", "segmented", "features"
                ]

                if any(k in lower for k in keys):
                    write_line(f, line)

            write_line(f)

        write_line(f, "=== END ===")

    print()
    print(f"Saved methodology facts at: {OUT}")


if __name__ == "__main__":
    main()
