from __future__ import annotations

from pathlib import Path
import torch
from riceleaf.config.loader import load_config, save_config
from riceleaf.data.datamodule import build_dataloaders
from riceleaf.models.registry import build_model
from riceleaf.training.seed import set_seed
from riceleaf.training.losses import build_loss
from riceleaf.training.optimizer import build_optimizer
from riceleaf.training.scheduler import build_scheduler
from riceleaf.training.trainer import train_model
from riceleaf.evaluation.metrics import compute_metrics, classification_report_df
from riceleaf.evaluation.confusion_matrix import save_confusion_matrix
from riceleaf.evaluation.inference_time import measure_inference_ms
from riceleaf.evaluation.model_size import count_parameters, estimate_model_size_mb
from riceleaf.evaluation.reports import save_predictions
from riceleaf.utils.paths import make_run_dir
from riceleaf.utils.serialization import save_json


def get_device(cfg):
    if cfg.project.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.project.device)


def predict(model, loader, device):
    model.eval()
    y_true, y_pred, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if "features" in batch:
                logits = model(images, batch["features"].to(device))
            else:
                logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            for rel, t, p, conf in zip(batch["relative_path"], labels.cpu().tolist(), preds.cpu().tolist(), probs.max(dim=1).values.cpu().tolist()):
                rows.append({"relative_path": rel, "true_label_idx": t, "pred_label_idx": p, "confidence": float(conf), "is_correct": int(t == p)})
    return y_true, y_pred, rows


def run_experiment(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    set_seed(int(cfg.project.seed))
    run_dir = make_run_dir(cfg.project.output_dir, cfg.experiment.id, cfg.experiment.name, cfg.project.seed)
    save_config(cfg, run_dir / "config.yml")
    device = get_device(cfg)
    loaders = build_dataloaders(cfg)
    model = build_model(cfg, feature_dim=loaders["feature_dim"]).to(device)
    criterion = build_loss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    train_info = train_model(model, loaders, criterion, optimizer, scheduler, cfg, device, run_dir)

    best_path = run_dir / "checkpoints" / "best_model.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred, pred_rows = predict(model, loaders["test"], device)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "experiment_id": cfg.experiment.id,
        "experiment_name": cfg.experiment.name,
        "seed": int(cfg.project.seed),
        "best_epoch": int(train_info["best_epoch"]),
        "inference_ms_per_image": measure_inference_ms(model, loaders["test"], device),
        "num_parameters": int(count_parameters(model)),
        "model_size_mb": estimate_model_size_mb(model),
    })
    save_json(metrics, run_dir / "metrics.json")
    class_names = list(cfg.data.class_names)
    classification_report_df(y_true, y_pred, class_names).to_csv(run_dir / "classification_report.csv")
    save_confusion_matrix(y_true, y_pred, class_names, run_dir / "confusion_matrix.png")
    save_predictions(pred_rows, run_dir / "predictions.csv")
    return run_dir
