from __future__ import annotations

import torch
import torch.nn as nn
from .mobilenet import MobileNetBackbone


class HybridMobileNetTexture(nn.Module):
    """Two-branch model: MobileNetV2 image embedding + handcrafted feature vector."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        texture_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0 for the hybrid model")
        self.backbone = MobileNetBackbone(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.texture_branch = nn.Sequential(
            nn.Linear(feature_dim, texture_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.out_dim + texture_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image, features):
        img_emb = self.backbone(image)
        tex_emb = self.texture_branch(features)
        fused = torch.cat([img_emb, tex_emb], dim=1)
        return self.classifier(fused)
