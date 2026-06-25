from __future__ import annotations

import cv2
import numpy as np
from .morphology import clean_mask


def lab_otsu_mask(
    rgb: np.ndarray,
    channel: str = "a",
    invert_mask: bool = False,
    kernel_size: int = 5,
    opening: bool = True,
    closing: bool = True,
    iterations: int = 1,
) -> np.ndarray:
    """Create a binary ROI mask using L*a*b* color conversion and Otsu thresholding.

    The `a` channel separates green-red information and the `b` channel separates blue-yellow.
    This is useful for leaf/lesion contrast under changing illumination.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    idx = {"l": 0, "a": 1, "b": 2}.get(channel.lower())
    if idx is None:
        raise ValueError("channel must be one of: l, a, b")
    ch = lab[:, :, idx]
    _, mask = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_mask:
        mask = 255 - mask
    mask = clean_mask(mask, kernel_size=kernel_size, opening=opening, closing=closing, iterations=iterations)
    return (mask > 0).astype(np.uint8)


def apply_mask(rgb: np.ndarray, mask: np.ndarray, background_value: int = 0) -> np.ndarray:
    out = np.full_like(rgb, fill_value=background_value)
    out[mask.astype(bool)] = rgb[mask.astype(bool)]
    return out


def segment_lab_otsu(rgb: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    mask = lab_otsu_mask(rgb, **kwargs)
    return apply_mask(rgb, mask), mask
