"""
utils/dataset.py
================
PyTorch Dataset classes for FaceForensics++ face crops.

Supports:
  - Binary classification (real vs fake)
  - Per-manipulation-type labels
  - Deterministic train/val/test splits using the official FF++ JSON splits
  - Both raw image mode (for EfficientNet) and DCT feature mode

Usage:
    from utils.dataset import FFPPDataset, DCTDataset, get_dataloaders
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.fft import dctn


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# ImageNet normalisation (used by EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Face crop size fed into the network
CROP_SIZE = 224

# Number of frames sampled per video during preprocessing
FRAMES_PER_VIDEO = 30


# --------------------------------------------------------------------------
# Image transforms
# --------------------------------------------------------------------------
def get_transforms(split: str) -> transforms.Compose:
    """
    Returns torchvision transforms for a given split.

    Args:
        split: one of 'train', 'val', 'test'

    Returns:
        Composed transform pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# --------------------------------------------------------------------------
# Helper: collect image paths + labels
# --------------------------------------------------------------------------
def _collect_samples(
    data_dir: str,
    manipulations: List[str],
    split: str,
    split_ids: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    """
    Walk data_dir and collect (image_path, label) pairs.

    Directory layout expected (output of 02_preprocess.py):
        data_dir/
          real/   <video_id>_<frame_idx>.jpg
          fake/
            Deepfakes/     <video_id>_<frame_idx>.jpg
            Face2Face/     ...
            FaceSwap/      ...
            NeuralTextures/...

    Args:
        data_dir:      Root directory of preprocessed face crops.
        manipulations: Which fake types to include.
        split:         'train', 'val', or 'test'.
        split_ids:     If provided, only include samples whose video_id is in this list.

    Returns:
        List of (path, label) tuples — label 0 = real, 1 = fake.
    """
    data_dir = Path(data_dir)
    samples: List[Tuple[str, int]] = []

    # --- Real faces ---
    real_dir = data_dir / split / "real"
    if real_dir.exists():
        for img_path in sorted(real_dir.glob("*.jpg")):
            vid_id = img_path.stem.rsplit("_", 1)[0]
            if split_ids is None or vid_id in split_ids:
                samples.append((str(img_path), 0))

    # --- Fake faces ---
    for manip in manipulations:
        fake_dir = data_dir / split / "fake" / manip
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.jpg")):
                vid_id = img_path.stem.rsplit("_", 1)[0]
                if split_ids is None or vid_id in split_ids:
                    samples.append((str(img_path), 1))

    return samples


# --------------------------------------------------------------------------
# FFPPDataset — for EfficientNet-B3 (raw image input)
# --------------------------------------------------------------------------
class FFPPDataset(Dataset):
    """
    FaceForensics++ dataset for CNN-based (spatial domain) classifiers.

    Returns (image_tensor, label) pairs where:
        image_tensor: FloatTensor of shape (3, 224, 224), ImageNet-normalised
        label:        0 = real,  1 = fake
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        manipulations: List[str] = None,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir:      Root of preprocessed face crop directory.
            split:         'train', 'val', or 'test'.
            manipulations: Fake types to include. Default = all four.
            transform:     Custom transform. Default = get_transforms(split).
            max_samples:   Cap dataset size (useful for quick runs / demo mode).
            seed:          Random seed for reproducibility.
        """
        if manipulations is None:
            manipulations = MANIPULATIONS

        self.data_dir      = data_dir
        self.split         = split
        self.manipulations = manipulations
        self.transform     = transform or get_transforms(split)

        self.samples = _collect_samples(data_dir, manipulations, split)

        # Balance real vs fake by up-sampling the minority class
        self.samples = self._balance(self.samples, seed=seed)

        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    # ------------------------------------------------------------------
    def _balance(
        self,
        samples: List[Tuple[str, int]],
        seed: int = 42
    ) -> List[Tuple[str, int]]:
        """
        Balance real vs fake samples by down-sampling the majority class.
        """
        real  = [s for s in samples if s[1] == 0]
        fake  = [s for s in samples if s[1] == 1]
        n     = min(len(real), len(fake))
        rng   = random.Random(seed)
        real  = rng.sample(real, n)
        fake  = rng.sample(fake, n)
        combined = real + fake
        rng.shuffle(combined)
        return combined

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    # ------------------------------------------------------------------
    @property
    def class_weights(self) -> torch.Tensor:
        """
        Returns inverse-frequency class weights for use with
        nn.CrossEntropyLoss(weight=...).
        """
        labels = np.array([s[1] for s in self.samples])
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------
# DCTDataset — for DCT frequency-domain classifiers
# --------------------------------------------------------------------------
class DCTDataset(Dataset):
    """
    FaceForensics++ dataset for DCT-based (frequency domain) classifiers.

    Extracts a fixed-length feature vector from each face image:
      1. Convert to YCbCr, take Y channel
      2. Divide into non-overlapping 8×8 blocks
      3. Apply 2D DCT to each block
      4. Flatten and concatenate DCT coefficients
      5. (Optionally) retain only the top-K zig-zag-ordered coefficients

    Returns (feature_vector, label) pairs.
    """

    # Zig-zag scan order indices for an 8×8 block (standard JPEG ordering)
    ZIGZAG_8x8 = [
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        manipulations: List[str] = None,
        img_size: int = 128,
        top_k_coeffs: int = 32,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir:      Root of preprocessed face crop directory.
            split:         'train', 'val', or 'test'.
            manipulations: Fake types to include. Default = all four.
            img_size:      Resize images to this square size before DCT.
                           Must be divisible by 8. Default = 128.
            top_k_coeffs:  Keep only the top-K DCT coefficients per 8×8 block
                           (in zig-zag order). Default = 32.
            max_samples:   Cap dataset size.
            seed:          Random seed.
        """
        assert img_size % 8 == 0, "img_size must be divisible by 8"

        if manipulations is None:
            manipulations = MANIPULATIONS

        self.data_dir      = data_dir
        self.split         = split
        self.manipulations = manipulations
        self.img_size      = img_size
        self.top_k_coeffs  = top_k_coeffs

        # Number of 8×8 blocks
        n_blocks = (img_size // 8) ** 2
        self.feature_dim = n_blocks * top_k_coeffs

        self.samples = _collect_samples(data_dir, manipulations, split)
        self.samples = self._balance(self.samples, seed=seed)

        if max_samples is not None:
            rng = random.Random(seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    # ------------------------------------------------------------------
    def _balance(self, samples, seed=42):
        real  = [s for s in samples if s[1] == 0]
        fake  = [s for s in samples if s[1] == 1]
        n     = min(len(real), len(fake))
        rng   = random.Random(seed)
        real  = rng.sample(real, n)
        fake  = rng.sample(fake, n)
        combined = real + fake
        rng.shuffle(combined)
        return combined

    # ------------------------------------------------------------------
    def _extract_dct_features(self, img_path: str) -> np.ndarray:
        """
        Extract DCT feature vector from an image.

        Steps:
          1. Read image → resize → convert to YCbCr
          2. Take Y (luminance) channel — most discriminative for forgery
          3. Divide into 8×8 blocks
          4. 2D DCT on each block
          5. Flatten top-K zig-zag coefficients per block
          6. Concatenate all blocks → 1-D feature vector

        Returns:
            np.ndarray of shape (feature_dim,), float32.
        """
        # Load and resize
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        img = cv2.resize(img, (self.img_size, self.img_size))
        # BGR → YCrCb, take Y channel
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)

        # Normalize to [-128, 127] as in JPEG DCT
        y -= 128.0

        h, w = self.img_size, self.img_size
        features = []

        for row in range(0, h, 8):
            for col in range(0, w, 8):
                block = y[row:row+8, col:col+8]
                # 2D DCT (scipy implementation, norm='ortho' matches JPEG standard)
                dct_block = dctn(block, norm='ortho')
                # Flatten in zig-zag order
                flat = dct_block.flatten()
                zigzag_coeffs = flat[self.ZIGZAG_8x8[:self.top_k_coeffs]]
                features.append(zigzag_coeffs)

        return np.concatenate(features, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img_path, label = self.samples[idx]
        features = self._extract_dct_features(img_path)
        return features, label

    # ------------------------------------------------------------------
    def get_all_features_labels(
        self,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for the entire dataset in one call.

        Useful for sklearn classifiers that expect an (N, D) matrix.

        Returns:
            X: np.ndarray of shape (N, feature_dim)
            y: np.ndarray of shape (N,)
        """
        X_list, y_list = [], []
        iterator = enumerate(self.samples)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(enumerate(self.samples),
                            total=len(self.samples),
                            desc=f"Extracting DCT [{self.split}]")

        for idx, (img_path, label) in iterator:
            feat = self._extract_dct_features(img_path)
            X_list.append(feat)
            y_list.append(label)

        return np.stack(X_list), np.array(y_list, dtype=np.int32)


# --------------------------------------------------------------------------
# Convenience function: get DataLoaders for EfficientNet
# --------------------------------------------------------------------------
def get_dataloaders(
    data_dir: str,
    manipulations: List[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Returns a dict of {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    for use with the EfficientNet-B3 trainer.

    Args:
        data_dir:      Root of preprocessed face crops.
        manipulations: Which manipulation types to include.
        batch_size:    Mini-batch size.
        num_workers:   DataLoader worker processes.
        max_samples:   Cap per split (useful for fast experiments).
        seed:          Random seed for dataset shuffling.

    Returns:
        Dictionary mapping split names to DataLoaders.
    """
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = FFPPDataset(
            data_dir=data_dir,
            split=split,
            manipulations=manipulations,
            max_samples=max_samples,
            seed=seed
        )
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train")
        )
    return dataloaders


# --------------------------------------------------------------------------
# Demo / sanity-check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Dataset module loaded OK.")
    print(f"  MANIPULATIONS  : {MANIPULATIONS}")
    print(f"  CROP_SIZE      : {CROP_SIZE}")
    print(f"  FRAMES_PER_VID : {FRAMES_PER_VIDEO}")

    # DCT feature dimension sanity check
    img_size, top_k = 128, 32
    n_blocks = (img_size // 8) ** 2
    feat_dim = n_blocks * top_k
    print(f"\nDCT feature dim for img_size={img_size}, top_k={top_k}: {feat_dim}")
