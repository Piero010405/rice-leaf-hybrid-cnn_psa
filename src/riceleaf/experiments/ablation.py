from __future__ import annotations

import pandas as pd


def summarize_ablation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact ablation table from experiment metrics."""
    cols = [c for c in ["experiment_id", "experiment_name", "accuracy", "macro_f1", "inference_ms_per_image"] if c in results_df.columns]
    return results_df[cols].copy()
