from __future__ import annotations

import tempfile
from pathlib import Path
import torch


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    torch.save(model.state_dict(), path)
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    Path(path).unlink(missing_ok=True)
    return float(size_mb)
