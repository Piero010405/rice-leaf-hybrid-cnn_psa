import importlib.util
import pytest
import numpy as np
from riceleaf.features.glcm import extract_glcm_features


def test_glcm_non_empty():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert len(extract_glcm_features(img)) > 0


@pytest.mark.skipif(importlib.util.find_spec("pywt") is None, reason="PyWavelets not installed")
def test_dwt_non_empty():
    from riceleaf.features.dwt import extract_dwt_features
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert len(extract_dwt_features(img)) > 0
