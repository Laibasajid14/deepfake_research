"""
utils/metrics.py
================
Evaluation helpers for deepfake detection experiments.

Provides:
  - compute_metrics()       — accuracy, AUC, F1, EER
  - confusion_matrix_fig()  — publication-ready confusion matrix heatmap
  - cross_manip_table()     — cross-manipulation generalisation matrix
  - print_results_table()   — console-friendly summary
  - save_results_csv()      — persist metrics to CSV
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    average_precision_score,
)


# --------------------------------------------------------------------------
# Core metric computation
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true:  Ground-truth labels (0 = real, 1 = fake). Shape (N,).
        y_pred:  Hard predictions (0 or 1). Shape (N,).
        y_score: Probability scores for the positive class (optional).
                 Required for AUC and EER computation. Shape (N,).

    Returns:
        Dictionary with keys:
            accuracy, precision, recall, f1, auc, eer, ap
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)

    metrics: Dict[str, float] = {}

    metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))

    if y_score is not None:
        y_score = np.asarray(y_score)
        # AUC-ROC
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auc"] = float("nan")

        # Average Precision (AUC-PR)
        try:
            metrics["ap"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["ap"] = float("nan")

        # Equal Error Rate (EER) — used in biometric literature
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        metrics["eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    else:
        metrics["auc"] = float("nan")
        metrics["ap"]  = float("nan")
        metrics["eer"] = float("nan")

    return metrics


# --------------------------------------------------------------------------
# Confusion matrix figure
# --------------------------------------------------------------------------
def confusion_matrix_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (5, 4),
) -> plt.Figure:
    """
    Create a publication-quality confusion matrix heatmap.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        title:       Plot title.
        class_names: Names for each class. Default = ['Real', 'Fake'].
        save_path:   If provided, save figure to this path.
        figsize:     Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    if class_names is None:
        class_names = ["Real", "Fake"]

    cm = confusion_matrix(y_true, y_pred)
    # Normalise to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --------------------------------------------------------------------------
# ROC curve figure
# --------------------------------------------------------------------------
def roc_curve_fig(
    results_dict: Dict[str, Dict],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        results_dict: {model_name: {'y_true': ..., 'y_score': ...}}
        title:        Plot title.
        save_path:    Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, data) in enumerate(results_dict.items()):
        y_true  = np.asarray(data["y_true"])
        y_score = np.asarray(data["y_score"])
        try:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr,
                    color=colors[i % len(colors)],
                    lw=2,
                    label=f"{name} (AUC = {auc:.3f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --------------------------------------------------------------------------
# Cross-manipulation table
# --------------------------------------------------------------------------
def cross_manip_table_fig(
    matrix: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Cross-Manipulation Generalisation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """
    Visualise a cross-manipulation generalisation matrix as a heatmap.

    Args:
        matrix:    Nested dict {train_manip: {test_manip: metric_value}}.
        metric:    Name of the metric being visualised (for annotation).
        title:     Plot title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure object.
    """
    manips = sorted(matrix.keys())
    data   = np.array([[matrix[t][e] for e in manips] for t in manips])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5, vmax=1.0,
        xticklabels=manips,
        yticklabels=manips,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric.upper()},
    )
    ax.set_xlabel("Tested On →", fontsize=11)
    ax.set_ylabel("← Trained On", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# --------------------------------------------------------------------------
# Console table printer
# --------------------------------------------------------------------------
def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted results table to stdout.

    Args:
        results: {experiment_name: metrics_dict}
    """
    header = f"{'Experiment':<35} {'Acc':>7} {'AUC':>7} {'F1':>7} {'EER':>7} {'AP':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        acc = f"{m.get('accuracy', float('nan')):.4f}"
        auc = f"{m.get('auc',      float('nan')):.4f}"
        f1  = f"{m.get('f1',       float('nan')):.4f}"
        eer = f"{m.get('eer',      float('nan')):.4f}"
        ap  = f"{m.get('ap',       float('nan')):.4f}"
        print(f"{name:<35} {acc:>7} {auc:>7} {f1:>7} {eer:>7} {ap:>7}")
    print("=" * len(header) + "\n")


# --------------------------------------------------------------------------
# CSV persistence
# --------------------------------------------------------------------------
def save_results_csv(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    append: bool = False
) -> None:
    """
    Save results dictionary to a CSV file.

    Args:
        results:   {experiment_name: metrics_dict}
        save_path: Destination CSV path.
        append:    If True, append to existing file instead of overwriting.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "accuracy", "precision",
                  "recall", "f1", "auc", "ap", "eer"]
    mode = "a" if append else "w"
    with open(save_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or os.path.getsize(save_path) == 0:
            writer.writeheader()
        for name, m in results.items():
            row = {"experiment": name}
            for k in fieldnames[1:]:
                row[k] = f"{m.get(k, float('nan')):.4f}"
            writer.writerow(row)


# --------------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke-test with random predictions
    rng     = np.random.default_rng(42)
    y_true  = rng.integers(0, 2, size=200)
    y_score = rng.random(size=200)
    y_pred  = (y_score > 0.5).astype(int)

    m = compute_metrics(y_true, y_pred, y_score)
    print("Smoke-test metrics:", {k: f"{v:.4f}" for k, v in m.items()})

    fig = confusion_matrix_fig(y_true, y_pred, title="Smoke Test CM",
                               save_path="/tmp/test_cm.png")
    print("Confusion matrix saved to /tmp/test_cm.png")
