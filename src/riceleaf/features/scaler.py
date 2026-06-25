from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

META_COLS = {"image_id", "relative_path", "label", "label_idx"}


def fit_transform_save(train_csv: str | Path, val_csv: str | Path, test_csv: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)
    feature_cols = [c for c in train.columns if c not in META_COLS]
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    train.to_csv(out_dir / "train_features_scaled.csv", index=False)
    val.to_csv(out_dir / "val_features_scaled.csv", index=False)
    test.to_csv(out_dir / "test_features_scaled.csv", index=False)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, out_dir / "scaler.pkl")
