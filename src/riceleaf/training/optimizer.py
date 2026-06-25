from __future__ import annotations

import torch


def build_optimizer(model, cfg):
    opt = cfg.training.get("optimizer", "adamw").lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    params = [p for p in model.parameters() if p.requires_grad]
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")
