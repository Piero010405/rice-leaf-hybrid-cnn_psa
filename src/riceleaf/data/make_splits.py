from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from riceleaf.data.validation import list_images, filter_valid_images


def make_splits(
    raw_dir: str | Path,
    split_dir: str | Path,
    class_names: list[str],
    extensions: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    raw_dir = Path(raw_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(raw_dir, extensions)
    valid_images, invalid_images = filter_valid_images(all_images)
    rows = []
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for p in valid_images:
        label = p.parent.name
        if label not in class_to_idx:
            continue
        rows.append({
            "image_id": p.stem,
            "relative_path": str(p.relative_to(raw_dir)).replace("\\", "/"),
            "path": str(p),
            "label": label,
            "label_idx": class_to_idx[label],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No se encontraron imágenes válidas en {raw_dir}")

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df["label_idx"], random_state=seed
    )
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size_adjusted, stratify=temp_df["label_idx"], random_state=seed
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    df.to_csv(split_dir / "all_valid.csv", index=False)
    if invalid_images:
        pd.DataFrame({"path": [str(p) for p in invalid_images]}).to_csv(split_dir / "invalid_images.csv", index=False)


def split_summary(split_dir: str | Path) -> pd.DataFrame:
    rows = []
    for name in ["train", "val", "test"]:
        df = pd.read_csv(Path(split_dir) / f"{name}.csv")
        counts = df["label"].value_counts().to_dict()
        for label, n in counts.items():
            rows.append({"split": name, "label": label, "count": int(n)})
    return pd.DataFrame(rows)
