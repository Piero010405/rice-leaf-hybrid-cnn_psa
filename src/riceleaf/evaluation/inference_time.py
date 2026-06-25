from __future__ import annotations

import time
import torch


def measure_inference_ms(model, loader, device, max_batches: int = 20) -> float:
    model.eval()
    total_images = 0
    start = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            images = batch["image"].to(device)
            if "features" in batch:
                features = batch["features"].to(device)
                _ = model(images, features)
            else:
                _ = model(images)
            total_images += images.size(0)
    elapsed = time.perf_counter() - start
    return float((elapsed / max(total_images, 1)) * 1000.0)
