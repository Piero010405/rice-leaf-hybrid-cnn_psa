import numpy as np
from riceleaf.segmentation.lab_otsu import segment_lab_otsu


def test_segment_shape():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [100, 180, 80]
    seg, mask = segment_lab_otsu(img)
    assert seg.shape == img.shape
    assert mask.shape == img.shape[:2]
