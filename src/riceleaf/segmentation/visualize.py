from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_segmentation_panel(original: np.ndarray, mask: np.ndarray, segmented: np.ndarray, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(segmented)
    axes[2].set_title("Segmented")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
