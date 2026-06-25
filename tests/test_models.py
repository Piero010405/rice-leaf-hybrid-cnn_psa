import torch
from riceleaf.models.mobilenet import build_mobilenetv2


def test_mobilenet_forward():
    model = build_mobilenetv2(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)
