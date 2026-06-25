from __future__ import annotations

import numpy as np


def bounding_box_from_mask(mask: np.ndarray, padding: int = 5) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, w)
    y2 = min(int(ys.max()) + padding + 1, h)
    return x1, y1, x2, y2


def crop_roi(image: np.ndarray, mask: np.ndarray, padding: int = 5) -> np.ndarray:
    bbox = bounding_box_from_mask(mask, padding=padding)
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]
