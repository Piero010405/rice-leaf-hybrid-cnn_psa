from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def save_bar_chart(labels, values, title: str, ylabel: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
