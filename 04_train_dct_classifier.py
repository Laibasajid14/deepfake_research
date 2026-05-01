"""
04_train_dct_classifier.py
==========================
Train SVM classifier on DCT frequency-domain features for deepfake detection.

Supports:
  - Block-wise DCT feature extraction
  - SVM with RBF kernel and probability estimates
  - Feature scaling with StandardScaler
  - Per-manipulation training
  - Demo mode for quick testing

Usage:
    python 04_train_dct_classifier.py \
        --data_dir data/faces \
        --manip all \
        --epochs 10 \
        --batch_size 32 \
        --output_dir models/dct_svm
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import List

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.dataset import DCTDataset, MANIPULATIONS


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Set random seeds
    np.random.seed(42)

    # Manipulations
    if args.manip == "all":
        manipulations = MANIPULATIONS
    else:
        manipulations = [args.manip]

    # Demo mode: limit samples
    max_samples = 500 if args.demo_mode else None

    # Load training data
    train_dataset = DCTDataset(
        data_dir=args.data_dir,
        split="train",
        manipulations=manipulations,
        max_samples=max_samples,
        seed=42
    )
    X_train, y_train = train_dataset.get_all_features_labels(verbose=True)

    # Load validation data for early stopping (SVM doesn't have epochs, but we can use val for tuning)
    val_dataset = DCTDataset(
        data_dir=args.data_dir,
        split="val",
        manipulations=manipulations,
        max_samples=max_samples,
        seed=42
    )
    X_val, y_val = val_dataset.get_all_features_labels(verbose=True)

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Feature dimension: {X_train.shape[1]}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SVM model
    # Use RBF kernel, probability=True for AUC computation
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)

    print("Training SVM...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = model.predict(X_val_scaled)
    val_acc = np.mean(val_pred == y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model and scaler
    model_path = output_dir / "svm_model.pkl"
    scaler_path = output_dir / "scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("Training complete.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM on DCT features for deepfake detection")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed face crops (e.g., data/faces)")
    parser.add_argument("--manip", type=str, default="all",
                        choices=["all"] + MANIPULATIONS,
                        help="Manipulation type(s) to train on")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Ignored for SVM (kept for consistency)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Ignored for SVM (kept for consistency)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and scaler")
    parser.add_argument("--demo_mode", action="store_true",
                        help="Run in demo mode (small dataset)")

    args = parser.parse_args()
    main(args)