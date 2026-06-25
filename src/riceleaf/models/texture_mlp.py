from __future__ import annotations

import torch.nn as nn


class TextureMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int] | tuple[int, ...] = (256, 128), dropout: float = 0.3):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_d, out_d), nn.ReLU(inplace=True), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)
