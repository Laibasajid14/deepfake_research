"""
05_evaluate.py
==============
Evaluate trained models on FaceForensics++ test sets.

Supports:
  - In-distribution evaluation (train/test on same manipulation)
  - Cross-manipulation generalization (train on A, test on B)
  - Comprehensive metrics: accuracy, AUC, F1, EER, AP
  - Confusion matrix figures
  - CSV exports for paper tables

Outputs:
  - results/main_results.csv     — Table 2 in paper
  - results/cross_manip.csv      — Table 3 in paper
  - results/figures/             — Confusion matrices
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

from utils.dataset import DCTDataset, FFPPDataset, MANIPULATIONS, get_dataloaders
from utils.metrics import compute_metrics, confusion_matrix_fig, save_results_csv


# --------------------------------------------------------------------------
# Model loading utilities
# --------------------------------------------------------------------------
class EfficientNetB3Binary(nn.Module):
    """Same as in training script."""
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


def load_efficientnet_model(model_dir: str, device: torch.device) -> nn.Module:
    """Load EfficientNet model from checkpoint."""
    model = EfficientNetB3Binary()
    checkpoint_path = Path(model_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_dct_model(model_dir: str) -> Tuple[Any, Any]:
    """Load SVM model and scaler."""
    model_path = Path(model_dir) / "svm_model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# --------------------------------------------------------------------------
# Evaluation functions
# --------------------------------------------------------------------------
def evaluate_efficientnet(
    model: nn.Module,
    dataset: FFPPDataset,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate EfficientNet on a dataset.
    Returns y_true, y_pred, y_score.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probs > 0.5).cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_score)


def evaluate_dct(
    model: Any,
    scaler: Any,
    dataset: DCTDataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate DCT SVM on a dataset.
    Returns y_true, y_pred, y_score.
    """
    X, y_true = dataset.get_all_features_labels(verbose=False)
    X_scaled = scaler.transform(X)

    y_score = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    return y_true, y_pred, y_score


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Demo mode: limit samples
    max_samples = 200 if args.demo_mode else None

    # Output directories
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # All manipulations
    manips = MANIPULATIONS

    # Results storage
    main_results = {}  # {method_manip: metrics}
    cross_results = {}  # {train_manip: {test_manip: {method: metrics}}}

    print("Starting evaluation...")

    # For each method
    for method in ["efficientnet", "dct"]:
        print(f"\nEvaluating {method.upper()}...")

        # For each train manipulation
        for train_manip in manips:
            print(f"  Loading {method} model trained on {train_manip}")

            # Load model
            if method == "efficientnet":
                model_dir = Path(args.models_dir) / "efficientnet" / train_manip
                try:
                    model = load_efficientnet_model(str(model_dir), device)
                except FileNotFoundError:
                    print(f"    Skipping {method} {train_manip}: model not found")
                    continue
            else:  # dct
                model_dir = Path(args.models_dir) / "dct_svm" / train_manip
                try:
                    model, scaler = load_dct_model(str(model_dir))
                except FileNotFoundError:
                    print(f"    Skipping {method} {train_manip}: model not found")
                    continue

            # For each test manipulation
            for test_manip in manips:
                print(f"    Testing on {test_manip}")

                # Load test dataset
                if method == "efficientnet":
                    test_dataset = FFPPDataset(
                        data_dir=args.data_dir,
                        split="test",
                        manipulations=[test_manip],
                        max_samples=max_samples,
                        seed=42
                    )
                    y_true, y_pred, y_score = evaluate_efficientnet(
                        model, test_dataset, device, batch_size=args.batch_size
                    )
                else:  # dct
                    test_dataset = DCTDataset(
                        data_dir=args.data_dir,
                        split="test",
                        manipulations=[test_manip],
                        max_samples=max_samples,
                        seed=42
                    )
                    y_true, y_pred, y_score = evaluate_dct(model, scaler, test_dataset)

                # Compute metrics
                metrics = compute_metrics(y_true, y_pred, y_score)

                # Store results
                exp_name = f"{method}_{train_manip}_on_{test_manip}"
                if train_manip == test_manip:
                    main_results[exp_name] = metrics

                if train_manip not in cross_results:
                    cross_results[train_manip] = {}
                if test_manip not in cross_results[train_manip]:
                    cross_results[train_manip][test_manip] = {}
                cross_results[train_manip][test_manip][method] = metrics

                # Confusion matrix
                cm_path = figures_dir / f"cm_{method}_{train_manip}_on_{test_manip}.png"
                confusion_matrix_fig(
                    y_true, y_pred,
                    title=f"{method.upper()} ({train_manip} → {test_manip})",
                    save_path=str(cm_path)
                )

    # Save main results CSV
    save_results_csv(main_results, results_dir / "main_results.csv")

    # Save cross-manipulation CSV
    # Flatten to {train_test_method: metrics}
    cross_flat = {}
    for train_manip, test_dict in cross_results.items():
        for test_manip, method_dict in test_dict.items():
            for method, metrics in method_dict.items():
                key = f"{method}_{train_manip}_on_{test_manip}"
                cross_flat[key] = metrics
    save_results_csv(cross_flat, results_dir / "cross_manip.csv")

    print(f"\nEvaluation complete.")
    print(f"Main results saved to {results_dir / 'main_results.csv'}")
    print(f"Cross-manip results saved to {results_dir / 'cross_manip.csv'}")
    print(f"Confusion matrices saved to {figures_dir}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed face crops (e.g., data/faces)")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory containing trained models")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for EfficientNet evaluation")
    parser.add_argument("--demo_mode", action="store_true",
                        help="Run in demo mode (small datasets)")

    args = parser.parse_args()
    main(args)