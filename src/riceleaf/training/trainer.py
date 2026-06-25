from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from riceleaf.evaluation.metrics import compute_metrics
from riceleaf.training.callbacks import EarlyStopping


def _forward(model, batch, device):
    images = batch["image"].to(device)
    if "features" in batch:
        features = batch["features"].to(device)
        return model(images, features)
    return model(images)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []
    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        labels = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch, device)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def train_model(model, loaders, criterion, optimizer, scheduler, cfg, device, run_dir: str | Path):
    run_dir = Path(run_dir)
    history = []
    stopper = EarlyStopping(cfg.training.early_stopping_patience, cfg.training.monitor_mode)
    best_metric = None
    best_epoch = -1
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        if scheduler is not None:
            if cfg.training.get("scheduler", "cosine") == "plateau":
                scheduler.step(val_metrics["macro_f1"])
            else:
                scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        monitor_key = cfg.training.monitor_metric
        value = row[monitor_key]
        improved = best_metric is None or (cfg.training.monitor_mode == "max" and value > best_metric) or (cfg.training.monitor_mode == "min" and value < best_metric)
        if improved:
            best_metric = value
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best_model.pt")
        torch.save(model.state_dict(), run_dir / "checkpoints" / "last_model.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if stopper.step(value):
            break
    return {"best_metric": float(best_metric), "best_epoch": best_epoch, "history": history}
