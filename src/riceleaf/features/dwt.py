from __future__ import annotations

import numpy as np
import pywt
from skimage.color import rgb2gray


def _entropy(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    energy = arr ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0
    p = energy / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _stats(prefix: str, arr: np.ndarray, stats: list[str] | tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "mean" in stats:
        out[f"{prefix}_mean"] = float(np.mean(arr))
    if "std" in stats:
        out[f"{prefix}_std"] = float(np.std(arr))
    if "energy" in stats:
        out[f"{prefix}_energy"] = float(np.sum(np.asarray(arr, dtype=np.float64) ** 2))
    if "entropy" in stats:
        out[f"{prefix}_entropy"] = _entropy(arr)
    return out


def extract_dwt_features(
    rgb: np.ndarray,
    wavelet: str = "db2",
    level: int = 2,
    stats: list[str] | tuple[str, ...] = ("mean", "std", "energy", "entropy"),
) -> dict[str, float]:
    """Extract multiscale wavelet features from an RGB image."""
    gray = rgb2gray(rgb).astype(np.float32)
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
    features: dict[str, float] = {}
    cA = coeffs[0]
    features.update(_stats("dwt_cA", cA, stats))
    for level_idx, detail in enumerate(coeffs[1:], start=1):
        cH, cV, cD = detail
        features.update(_stats(f"dwt_L{level_idx}_cH", cH, stats))
        features.update(_stats(f"dwt_L{level_idx}_cV", cV, stats))
        features.update(_stats(f"dwt_L{level_idx}_cD", cD, stats))
    return features
