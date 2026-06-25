from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_predictions(rows: list[dict], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
