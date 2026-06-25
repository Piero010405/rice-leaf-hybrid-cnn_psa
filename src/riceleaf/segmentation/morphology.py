from __future__ import annotations

import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5, opening: bool = True, closing: bool = True, iterations: int = 1) -> np.ndarray:
    """Clean a binary mask using morphological opening/closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = mask.astype(np.uint8)
    if opening:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if closing:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return out
