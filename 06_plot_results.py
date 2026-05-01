"""
06_plot_results.py
==================
Generate publication-ready plots from evaluation results.

Creates:
  - Bar chart: EfficientNet vs DCT accuracy per manipulation
  - Heatmap: Cross-manipulation generalization matrix
  - Saves to results/figures/
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.metrics import cross_manip_table_fig


# --------------------------------------------------------------------------
# Plot functions
# --------------------------------------------------------------------------
def plot_accuracy_bar_chart(
    main_results: pd.DataFrame,
    save_path: str
) -> None:
    """
    Bar chart comparing EfficientNet and DCT accuracy per manipulation.
    """
    # Extract accuracies
    manips = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    efficientnet_accs = []
    dct_accs = []

    for manip in manips:
        eff_key = f"efficientnet_{manip}_on_{manip}"
        dct_key = f"dct_{manip}_on_{manip}"

        eff_acc = main_results.loc[main_results["experiment"] == eff_key, "accuracy"].values
        dct_acc = main_results.loc[main_results["experiment"] == dct_key, "accuracy"].values

        efficientnet_accs.append(eff_acc[0] if len(eff_acc) > 0 else 0.5)
        dct_accs.append(dct_acc[0] if len(dct_acc) > 0 else 0.5)

    # Plot
    x = np.arange(len(manips))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, efficientnet_accs, width, label="EfficientNet-B3", color="#1f77b4")
    ax.bar(x + width/2, dct_accs, width, label="DCT + SVM", color="#ff7f0e")

    ax.set_xlabel("Manipulation Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("In-Distribution Accuracy: EfficientNet vs DCT")
    ax.set_xticks(x)
    ax.set_xticklabels(manips, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_manip_heatmap(
    cross_results: pd.DataFrame,
    method: str,
    save_path: str
) -> None:
    """
    Heatmap of cross-manipulation generalization for a given method.
    """
    manips = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    # Build matrix
    matrix = {}
    for train_manip in manips:
        matrix[train_manip] = {}
        for test_manip in manips:
            key = f"{method}_{train_manip}_on_{test_manip}"
            acc = cross_results.loc[cross_results["experiment"] == key, "accuracy"].values
            matrix[train_manip][test_manip] = acc[0] if len(acc) > 0 else 0.5

    # Plot
    cross_manip_table_fig(
        matrix,
        metric="accuracy",
        title=f"Cross-Manipulation Generalization ({method.upper()})",
        save_path=save_path
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    main_csv = results_dir / "main_results.csv"
    cross_csv = results_dir / "cross_manip.csv"

    if not main_csv.exists() or not cross_csv.exists():
        print("Error: Results CSVs not found. Run 05_evaluate.py first.")
        return

    main_results = pd.read_csv(main_csv)
    cross_results = pd.read_csv(cross_csv)

    # Plot bar chart
    bar_path = figures_dir / "accuracy_comparison.png"
    plot_accuracy_bar_chart(main_results, str(bar_path))
    print(f"Bar chart saved to {bar_path}")

    # Plot heatmaps
    for method in ["efficientnet", "dct"]:
        heatmap_path = figures_dir / f"cross_manip_{method}.png"
        plot_cross_manip_heatmap(cross_results, method, str(heatmap_path))
        print(f"Heatmap for {method} saved to {heatmap_path}")

    print("Plotting complete.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from evaluation results")
    # No args needed, reads from results/
    args = parser.parse_args()
    main(args)