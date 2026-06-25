from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash


SPLIT_DIR = Path("data/splits")
RAW_DIR = Path("data/raw/rice_leaf_disease")


def get_image_path(row):
    if "path" in row and pd.notna(row["path"]):
        return Path(row["path"])

    if "relative_path" in row and pd.notna(row["relative_path"]):
        return RAW_DIR / row["relative_path"]

    raise ValueError("El CSV no tiene columna path ni relative_path.")


def main():
    rows = []

    for split_name in ["train", "val", "test"]:
        df = pd.read_csv(SPLIT_DIR / f"{split_name}.csv")

        for _, r in df.iterrows():
            path = get_image_path(r)

            try:
                img = Image.open(path).convert("RGB")
                phash = str(imagehash.phash(img))

                rows.append({
                    "split": split_name,
                    "label": r["label"],
                    "path": str(path),
                    "phash": phash,
                })

            except Exception as e:
                print("Error:", path, e)

    all_df = pd.DataFrame(rows)

    dups = all_df.groupby("phash").filter(lambda x: x["split"].nunique() > 1)

    print("Total imágenes:", len(all_df))
    print("pHashes únicos:", all_df["phash"].nunique())
    print("Posibles duplicados visuales exactos por pHash entre splits:", len(dups))

    if len(dups) > 0:
        print(dups.sort_values("phash").head(100).to_string(index=False))
    else:
        print("OK: no hay duplicados visuales exactos por pHash entre splits.")


if __name__ == "__main__":
    main()
    