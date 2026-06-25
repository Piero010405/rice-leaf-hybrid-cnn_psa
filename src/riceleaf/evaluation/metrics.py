from __future__ import annotations

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def classification_report_df(y_true, y_pred, class_names: list[str]):
    import pandas as pd
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
