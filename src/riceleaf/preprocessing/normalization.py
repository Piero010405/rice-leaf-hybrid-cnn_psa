from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD
