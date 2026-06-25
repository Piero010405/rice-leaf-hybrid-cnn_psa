from __future__ import annotations

import torch


def build_scheduler(optimizer, cfg):
    name = cfg.training.get("scheduler", "cosine")
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    raise ValueError(f"Unknown scheduler: {name}")
