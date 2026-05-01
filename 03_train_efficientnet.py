"""
03_train_efficientnet.py
========================
Train EfficientNet-B3 for deepfake detection on FaceForensics++.

Supports:
  - Binary classification (real vs fake)
  - Per-manipulation training (for cross-manipulation experiments)
  - Early stopping and model checkpointing
  - Training logs saved to CSV
  - Demo mode for quick testing

Usage:
    python 03_train_efficientnet.py \
        --data_dir data/faces \
        --manip all \
        --epochs 10 \
        --batch_size 32 \
        --output_dir models/efficientnet
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights
import numpy as np

from utils.dataset import get_dataloaders, MANIPULATIONS


# --------------------------------------------------------------------------
# Model: EfficientNet-B3 with binary classifier head
# --------------------------------------------------------------------------
class EfficientNetB3Binary(nn.Module):
    """
    EfficientNet-B3 adapted for binary classification.
    Replaces the final fully-connected layer with a single output.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load EfficientNet-B3
        self.backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        # Replace classifier: original is (1000,) → (1,) for binary
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# --------------------------------------------------------------------------
# Training utilities
# --------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    Returns average loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if i % 50 == 0:
            print(f"Training batch {i}/{len(dataloader)}")

        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate the model.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model state dict."""
    torch.save(model.state_dict(), path)


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Manipulations
    if args.manip == "all":
        manipulations = MANIPULATIONS
    else:
        manipulations = [args.manip]

    # Demo mode: limit samples
    max_samples = 1000 if args.demo_mode else None

    # DataLoaders
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        manipulations=manipulations,
        batch_size=args.batch_size,
        max_samples=max_samples,
        seed=42
    )

    # Model
    model = EfficientNetB3Binary(pretrained=not args.demo_mode)
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # Early stopping
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Training EfficientNet-B3 on manipulations: {manipulations}")
    print(f"Output dir: {output_dir}")
    print(f"Demo mode: {args.demo_mode}")

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, dataloaders["val"], criterion, device)

        # Log
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"]
            ])

        # Checkpoint best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training complete. Best model saved.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 for deepfake detection")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed face crops (e.g., data/faces)")
    parser.add_argument("--manip", type=str, default="all",
                        choices=["all"] + MANIPULATIONS,
                        help="Manipulation type(s) to train on")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and logs")
    parser.add_argument("--demo_mode", action="store_true",
                        help="Run in demo mode (small dataset, no pretrained weights)")

    args = parser.parse_args()
    main(args)