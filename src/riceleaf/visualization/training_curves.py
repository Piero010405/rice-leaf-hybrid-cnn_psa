from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_training_curves(history_csv: str | Path, out_dir: str | Path) -> None:
    df = pd.read_csv(history_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["loss", "macro_f1", "accuracy"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for split in ["train", "val"]:
            col = f"{split}_{metric}"
            if col in df:
                ax.plot(df["epoch"], df[col], label=col)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"curve_{metric}.png", dpi=160)
        plt.close(fig)
