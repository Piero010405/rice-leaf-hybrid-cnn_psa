from __future__ import annotations

from omegaconf import DictConfig
from .mobilenet import build_mobilenetv2
from .hybrid import HybridMobileNetTexture


def build_model(cfg: DictConfig, feature_dim: int = 0):
    m = cfg.model
    if m.type == "mobilenetv2":
        return build_mobilenetv2(
            num_classes=m.num_classes,
            pretrained=m.get("pretrained", True),
            freeze_backbone=m.get("freeze_backbone", False),
            dropout=m.get("dropout", 0.2),
        )
    if m.type == "hybrid_mobilenet_texture":
        return HybridMobileNetTexture(
            feature_dim=feature_dim,
            num_classes=m.num_classes,
            pretrained=m.get("pretrained", True),
            freeze_backbone=m.get("freeze_backbone", False),
            texture_hidden_dim=m.get("texture_hidden_dim", 128),
            fusion_hidden_dim=m.get("fusion_hidden_dim", 256),
            dropout=m.get("dropout", 0.3),
        )
    raise ValueError(f"Unknown model type: {m.type}")
