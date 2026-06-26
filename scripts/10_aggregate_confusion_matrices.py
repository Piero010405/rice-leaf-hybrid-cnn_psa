from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


RUNS_DIR = Path("runs")
OUT_DIR = Path("outputs/figures/aggregate_confusion_matrices")
TABLE_OUT_DIR = Path("outputs/tables")

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot",
    "Tungro",
]

EXPERIMENTS = {
    "E0_baseline_cnn": "E0 Baseline CNN",
    "E1_segmented_cnn": "E1 Segmented CNN",
    "E2_cnn_texture": "E2 CNN + Texture",
    "E3_proposed_segmented_cnn_texture": "E3 Proposed Hybrid",
}


def infer_seed(path: Path):
    m = re.search(r"_seed(\d+)", str(path))
    return int(m.group(1)) if m else None


def find_column(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def get_true_pred_columns(df: pd.DataFrame):
    true_candidates = [
        "true_label_idx",
        "y_true",
        "true",
        "true_label",
        "label_true",
        "label",
        "label_idx",
        "target",
        "target_idx",
        "actual",
        "actual_label",
    ]

    pred_candidates = [
        "pred_label_idx",
        "y_pred",
        "pred",
        "pred_label",
        "prediction",
        "predicted",
        "predicted_label",
        "pred_idx",
        "prediction_idx",
        "predicted_idx",
    ]

    true_col = find_column(df, true_candidates)
    pred_col = find_column(df, pred_candidates)

    if true_col is None or pred_col is None:
        raise ValueError(
            "No se pudieron detectar columnas true/pred. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    return true_col, pred_col


def normalize_labels(series: pd.Series):
    """
    Converts either integer class IDs or string class names to integer labels.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    return series.map(class_to_idx).astype(int).to_numpy()


def plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(cm)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    max_val = cm.max() if cm.size else 0
    threshold = max_val / 2 if max_val > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_rows = []

    for exp_dir_name, exp_label in EXPERIMENTS.items():
        exp_dir = RUNS_DIR / exp_dir_name

        if not exp_dir.exists():
            print(f"[WARN] No existe: {exp_dir}")
            continue

        pred_files = sorted(exp_dir.glob("*/predictions.csv"))

        if not pred_files:
            print(f"[WARN] No predictions.csv para {exp_dir_name}")
            continue

        y_true_all = []
        y_pred_all = []
        seeds = []

        for pred_file in pred_files:
            df = pd.read_csv(pred_file)
            true_col, pred_col = get_true_pred_columns(df)

            y_true = normalize_labels(df[true_col])
            y_pred = normalize_labels(df[pred_col])

            y_true_all.extend(y_true.tolist())
            y_pred_all.extend(y_pred.tolist())

            seed = infer_seed(pred_file)
            if seed is not None:
                seeds.append(seed)

            print(
                f"{exp_dir_name} | {pred_file} | rows={len(df)} | "
                f"true={true_col} | pred={pred_col}"
            )

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=list(range(len(CLASS_NAMES)))
        )

        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_csv = TABLE_OUT_DIR / f"aggregate_confusion_{exp_dir_name}.csv"
        cm_df.to_csv(cm_csv)

        fig_path = OUT_DIR / f"aggregate_confusion_{exp_dir_name}.png"
        plot_confusion(cm, exp_label, fig_path)

        total = cm.sum()
        correct = np.trace(cm)
        errors = total - correct

        summary_rows.append({
            "experiment": exp_dir_name,
            "label": exp_label,
            "n_prediction_files": len(pred_files),
            "seeds": ",".join(str(s) for s in sorted(seeds)),
            "total_predictions": int(total),
            "correct_predictions": int(correct),
            "errors": int(errors),
            "accuracy_from_aggregate_cm": correct / total if total > 0 else np.nan,
            "figure": str(fig_path),
            "csv": str(cm_csv),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = TABLE_OUT_DIR / "aggregate_confusion_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== AGGREGATE CONFUSION SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary at: {summary_path}")
    print(f"Saved figures at: {OUT_DIR}")


if __name__ == "__main__":
    main()