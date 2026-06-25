from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte


def extract_glcm_features(
    rgb: np.ndarray,
    distances: list[int] | tuple[int, ...] = (1, 2, 4),
    angles: list[float] | tuple[float, ...] = (0, np.pi/4, np.pi/2, 3*np.pi/4),
    levels: int = 32,
    properties: list[str] | tuple[str, ...] = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"),
) -> dict[str, float]:
    """Extract GLCM texture features from an RGB image."""
    gray = img_as_ubyte(rgb2gray(rgb))
    quantized = (gray / (256 / levels)).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    features: dict[str, float] = {}
    for prop in properties:
        vals = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(vals))
        features[f"glcm_{prop}_std"] = float(np.std(vals))
    return features
