from pathlib import Path
import random

import pandas as pd
from PIL import Image
import imagehash


RAW_DIR = Path("data/raw/rice_leaf_disease")
SPLIT_DIR = Path("data/splits")

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot",
    "Tungro",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_phash(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return str(imagehash.phash(img))


def collect_images() -> pd.DataFrame:
    rows = []
    seen_paths = set()

    for label in CLASS_NAMES:
        class_dir = RAW_DIR / label

        if not class_dir.exists():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            resolved = str(path.resolve()).lower()
            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)

            try:
                phash = compute_phash(path)
                relative_path = path.relative_to(RAW_DIR)

                rows.append({
                    "path": str(path),
                    "relative_path": str(relative_path),
                    "label": label,
                    "phash": phash,
                    "group_id": f"{label}__{phash}",
                })

            except Exception as e:
                print(f"[WARN] No se pudo leer {path}: {e}")

    return pd.DataFrame(rows)


def split_groups_by_label(df: pd.DataFrame):
    train_rows = []
    val_rows = []
    test_rows = []

    for label in CLASS_NAMES:
        df_label = df[df["label"] == label].copy()

        groups = []
        for _, group_df in df_label.groupby("group_id"):
            groups.append(group_df)

        random.shuffle(groups)

        total_images = sum(len(g) for g in groups)
        target_train = int(total_images * TRAIN_RATIO)
        target_val = int(total_images * VAL_RATIO)

        current_train = 0
        current_val = 0

        for group_df in groups:
            group_size = len(group_df)

            if current_train < target_train:
                train_rows.append(group_df)
                current_train += group_size
            elif current_val < target_val:
                val_rows.append(group_df)
                current_val += group_size
            else:
                test_rows.append(group_df)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    return train_df, val_df, test_df


def check_no_group_leakage(train_df, val_df, test_df):
    split_map = {}

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        for group_id in split_df["group_id"].unique():
            split_map.setdefault(group_id, set()).add(split_name)

    leaking = {g: s for g, s in split_map.items() if len(s) > 1}

    if leaking:
        print("[ERROR] Hay leakage de grupos entre splits:")
        for group_id, splits in list(leaking.items())[:20]:
            print(group_id, splits)
        raise RuntimeError("El split agrupado falló: hay group_id en más de un split.")

    print("OK: no hay leakage de group_id entre train/val/test.")


def save_split(df: pd.DataFrame, split_name: str):
    cols = ["path", "relative_path", "label", "phash", "group_id"]
    out = df[cols].copy()
    out.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False, encoding="utf-8")


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Recolectando imágenes y calculando pHash...")
    df = collect_images()

    print(f"Total imágenes: {len(df)}")
    print(f"pHashes únicos globales: {df['phash'].nunique()}")
    print(f"Grupos label+pHash únicos: {df['group_id'].nunique()}")

    if len(df) != 5932:
        print()
        print("[WARN] El total no es 5932. Revisa si hay archivos duplicados o carpetas adicionales en data/raw/rice_leaf_disease.")
        print()

    train_df, val_df, test_df = split_groups_by_label(df)

    check_no_group_leakage(train_df, val_df, test_df)

    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")

    summary = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])

    print()
    print("Resumen por split y clase:")
    print(summary.groupby(["split", "label"]).size().reset_index(name="count"))

    print()
    print("Resumen general:")
    print(summary.groupby("split").size().reset_index(name="count"))

    print()
    print(f"Splits guardados en: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
