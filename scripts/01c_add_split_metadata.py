from pathlib import Path
import pandas as pd

SPLIT_DIR = Path("data/splits")

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot",
    "Tungro",
]

CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_NAMES)}


def fix_split(split_name: str):
    path = SPLIT_DIR / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError(f"{path} no tiene columna label.")

    if "relative_path" not in df.columns:
        if "path" not in df.columns:
            raise ValueError(f"{path} no tiene relative_path ni path.")

        raw_root = Path("data/raw/rice_leaf_disease")
        df["relative_path"] = df["path"].apply(
            lambda p: str(Path(p).relative_to(raw_root))
            if str(p).startswith(str(raw_root))
            else str(Path(p))
        )

    df["label_idx"] = df["label"].map(CLASS_TO_IDX)

    if df["label_idx"].isna().any():
        bad_labels = df[df["label_idx"].isna()]["label"].unique()
        raise ValueError(f"Labels no reconocidos en {path}: {bad_labels}")

    df["label_idx"] = df["label_idx"].astype(int)

    df["image_id"] = df["relative_path"].apply(lambda p: Path(p).stem)

    ordered_cols = []

    for col in ["image_id", "path", "relative_path", "label", "label_idx", "phash", "group_id"]:
        if col in df.columns:
            ordered_cols.append(col)

    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    df.to_csv(path, index=False, encoding="utf-8")

    print(f"{split_name}: OK -> {path}")
    print(df[["image_id", "relative_path", "label", "label_idx"]].head())


def main():
    for split in ["train", "val", "test"]:
        fix_split(split)

    print()
    print("Columnas finales:")
    for split in ["train", "val", "test"]:
        df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
        print(split, df.columns.tolist(), len(df))


if __name__ == "__main__":
    main()
    