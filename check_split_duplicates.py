from pathlib import Path
import pandas as pd
import hashlib

SPLIT_DIR = Path("data/splits")

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []

for split_name in ["train", "val", "test"]:
    df = pd.read_csv(SPLIT_DIR / f"{split_name}.csv")
    for _, r in df.iterrows():
        path = Path(r["path"])
        rows.append({
            "split": split_name,
            "label": r["label"],
            "path": str(path),
            "hash": file_hash(path)
        })

all_df = pd.DataFrame(rows)

dups = all_df.groupby("hash").filter(lambda x: x["split"].nunique() > 1)

print("Total imágenes:", len(all_df))
print("Hashes únicos:", all_df["hash"].nunique())
print("Duplicados exactos entre splits:", len(dups))

if len(dups) > 0:
    print(dups.sort_values("hash").to_string(index=False))
else:
    print("OK: no hay duplicados exactos entre train/val/test.")