from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def collect_metrics(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for metrics_path in Path(runs_dir).rglob("metrics.json"):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        data["run_dir"] = str(metrics_path.parent)
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    preferred = ["experiment_id", "experiment_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "inference_ms_per_image", "num_parameters", "model_size_mb", "best_epoch", "run_dir"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].sort_values(["experiment_id", "macro_f1"], ascending=[True, False])


def save_comparison(runs_dir: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = collect_metrics(runs_dir)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
