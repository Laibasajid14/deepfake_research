"""
02_preprocess.py
================
Extract face crops from FaceForensics++ videos and organise them into
a directory structure suitable for FFPPDataset / DCTDataset.

Output layout:
    output_dir/
      train/
        real/      <vid_id>_<frame>.jpg
        fake/
          Deepfakes/      <vid_id>_<frame>.jpg
          Face2Face/
          FaceSwap/
          NeuralTextures/
      val/
        ...
      test/
        ...

Face detection uses MTCNN (facenet-pytorch).
Falls back to a simple Haar-cascade detector if MTCNN is unavailable.

Usage:
    # Full FF++ dataset
    python 02_preprocess.py \\
        --data_root data/FaceForensics++ \\
        --output_dir data/faces \\
        --frames_per_video 30 \\
        --img_size 224

    # Quick demo (generates synthetic faces — no FF++ needed)
    python 02_preprocess.py --demo_mode
"""

import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

# Default FF++ split JSON path (inside the downloaded dataset)
# SPLIT_JSON = "data/FaceForensics++/splits/train_val_test.json"
SPLIT_JSON = "data/FaceForensics++_C23/splits/train_val_test.json"

# Fallback: fixed split ratios if JSON not available
SPLIT_RATIOS = {"train": 0.72, "val": 0.08, "test": 0.20}
TOTAL_VIDEOS  = 1000  # FF++ has 1000 source videos


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FF++ face crop extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_root",       type=str, default="data/FaceForensics++",
                   help="Root directory of the downloaded FF++ dataset.")
    p.add_argument("--output_dir",      type=str, default="data/faces",
                   help="Where to save extracted face crops.")
    p.add_argument("--frames_per_video", type=int, default=30,
                   help="Number of frames to sample from each video.")
    p.add_argument("--img_size",        type=int, default=224,
                   help="Output face crop size (square, pixels).")
    p.add_argument("--margin",          type=float, default=0.3,
                   help="Face bounding box margin fraction.")
    p.add_argument("--split_json",      type=str, default=SPLIT_JSON,
                   help="Path to FF++ official split JSON.")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--demo_mode",       action="store_true",
                   help="Generate synthetic demo data (no FF++ download needed).")
    p.add_argument("--max_videos",      type=int, default=None,
                   help="Cap number of videos processed per category (for testing).")
    return p.parse_args()


# --------------------------------------------------------------------------
# Face detector — MTCNN preferred, Haar-cascade fallback
# --------------------------------------------------------------------------
class FaceDetector:
    """Wraps either MTCNN (preferred) or OpenCV Haar cascade."""

    def __init__(self):
        self.backend = None
        self._init_mtcnn()
        if self.backend is None:
            self._init_haar()

    def _init_mtcnn(self):
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
            self.backend = "mtcnn"
            print("[FaceDetector] Using MTCNN.")
        except ImportError:
            pass

    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        self.backend = "haar"
        print("[FaceDetector] MTCNN unavailable — using Haar cascade fallback.")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in a BGR frame.

        Returns:
            (x1, y1, x2, y2) bounding box, or None if no face found.
        """
        if self.backend == "mtcnn":
            return self._detect_mtcnn(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_mtcnn(self, frame_bgr):
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes, _ = self.mtcnn.detect(pil)
        if boxes is None or len(boxes) == 0:
            return None
        # Pick the largest box
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best  = boxes[int(np.argmax(areas))]
        return tuple(int(v) for v in best)   # x1, y1, x2, y2

    def _detect_haar(self, frame_bgr):
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # Return the largest detected face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return x, y, x + w, y + h


# --------------------------------------------------------------------------
# Frame + crop extraction helpers
# --------------------------------------------------------------------------
def sample_frames(
    video_path: str,
    n_frames: int,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Uniformly sample n_frames from a video.

    Args:
        video_path: Path to video file.
        n_frames:   Number of frames to sample.
        seed:       Random seed (for reproducibility).

    Returns:
        List of BGR frames (np.ndarray).
    """
    cap    = cv2.VideoCapture(str(video_path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = sorted(random.Random(seed).sample(
        range(total), min(n_frames, total)
    ))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def crop_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float,
    img_size: int
) -> np.ndarray:
    """
    Crop and resize a face from a frame given its bounding box.

    Args:
        frame:    Full BGR frame.
        bbox:     (x1, y1, x2, y2) face bounding box.
        margin:   Fractional margin added around the box.
        img_size: Output square size.

    Returns:
        Cropped and resized BGR face image.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)

    H, W = frame.shape[:2]
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(W, x2 + mx)
    y2 = min(H, y2 + my)

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (img_size, img_size))
    return crop


# --------------------------------------------------------------------------
# Load FF++ official splits (JSON)
# --------------------------------------------------------------------------
def load_splits(split_json: str) -> dict:
    """
    Load the official FF++ train/val/test video-ID splits.

    The JSON format is:
        {"train": [["001", "002"], ...], "val": [...], "test": [...]}
    Each entry is a pair [source_id, target_id].

    Returns:
        {"train": set_of_ids, "val": set_of_ids, "test": set_of_ids}
    """
    if not os.path.exists(split_json):
        print(f"[WARN] Split JSON not found at '{split_json}'.")
        print("  → Using automatic ratio-based splits.")
        return None

    with open(split_json) as f:
        raw = json.load(f)

    splits = {}
    for split, pairs in raw.items():
        ids = set()
        for pair in pairs:
            ids.add(pair[0])
            ids.add(pair[1])
        splits[split.lower()] = ids
    return splits


def assign_split_by_ratio(
    video_ids: List[str],
    ratios: dict = SPLIT_RATIOS,
    seed: int = 42
) -> dict:
    """
    Assign video IDs to splits by ratio when no JSON is available.

    Returns:
        {"train": set, "val": set, "test": set}
    """
    ids = sorted(video_ids)
    random.Random(seed).shuffle(ids)
    n     = len(ids)
    n_tr  = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": set(ids[:n_tr]),
        "val":   set(ids[n_tr:n_tr + n_val]),
        "test":  set(ids[n_tr + n_val:]),
    }


# --------------------------------------------------------------------------
# Main extraction pipeline
# --------------------------------------------------------------------------
def extract_category(
    video_dir: Path,
    output_base: Path,
    category: str,           # "real" or manipulation name
    splits: dict,
    detector: "FaceDetector",
    frames_per_video: int,
    img_size: int,
    margin: float,
    seed: int = 42,
    max_videos: Optional[int] = None,
) -> int:
    """
    Extract face crops from all videos in a category directory.

    Args:
        video_dir:        Directory containing .mp4 files.
        output_base:      Root output directory (data/faces).
        category:         "real" or e.g. "Deepfakes".
        splits:           {"train": set, "val": set, "test": set}
        detector:         FaceDetector instance.
        frames_per_video: Number of frames per video.
        img_size:         Crop output size.
        margin:           Face box margin.
        seed:             Random seed for frame sampling.
        max_videos:       Optional cap on number of videos.

    Returns:
        Total number of crops saved.
    """
    videos  = sorted(video_dir.glob("*.mp4"))
    if not videos:
        videos = sorted(video_dir.glob("*.avi"))
    if max_videos:
        videos = videos[:max_videos]

    if len(videos) == 0:
        print(f"  [WARN] No videos found in {video_dir}")
        return 0

    total_saved = 0
    for video_path in tqdm(videos, desc=f"  {category}", leave=False):
        vid_id = video_path.stem   # e.g. "000"

        # Determine split for this video
        split = "train"  # default
        for sp, id_set in splits.items():
            if vid_id in id_set:
                split = sp
                break

        # Output directory
        if category == "real":
            out_dir = output_base / split / "real"
        else:
            out_dir = output_base / split / "fake" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sample frames and extract faces
        frames = sample_frames(str(video_path), frames_per_video, seed=seed)
        for fi, frame in enumerate(frames):
            bbox = detector.detect(frame)
            if bbox is None:
                # Fallback: use centre crop if no face detected
                h, w = frame.shape[:2]
                m = min(h, w)
                bbox = ((w - m) // 2, (h - m) // 2,
                        (w + m) // 2, (h + m) // 2)

            crop     = crop_face(frame, bbox, margin, img_size)
            out_name = f"{vid_id}_{fi:04d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_saved += 1

    return total_saved


# --------------------------------------------------------------------------
# Demo mode — generate synthetic face-like crops (no FF++ required)
# --------------------------------------------------------------------------
def generate_demo_data(output_dir: str, img_size: int = 224, seed: int = 42):
    """
    Generate synthetic face-like images for pipeline testing.
    Real faces = uniform colour blobs.
    Fake faces = the same + a visible DCT-like checkerboard artifact.
    """
    print("\n[DEMO MODE] Generating synthetic face crops...")
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)

    manips = MANIPULATIONS
    splits = {"train": 300, "val": 60, "test": 80}  # samples per class per split

    for split, n in splits.items():
        # Real
        real_dir = output_dir / split / "real"
        real_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            base_color = rng.integers(100, 200, size=3).tolist()
            img = np.full((img_size, img_size, 3), base_color, dtype=np.uint8)
            # Add random noise
            noise = rng.integers(-15, 16,
                                  size=(img_size, img_size, 3)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Sketch a simple oval "face"
            cx, cy = img_size // 2, img_size // 2
            cv2.ellipse(img, (cx, cy), (img_size // 3, img_size // 2 + 10),
                        0, 0, 360, (200, 180, 160), -1)
            cv2.imwrite(str(real_dir / f"r{i:05d}_0000.jpg"), img)

        # Fake (one file per manipulation for variety)
        for manip in manips:
            fake_dir = output_dir / split / "fake" / manip
            fake_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                base_color = rng.integers(100, 200, size=3).tolist()
                img = np.full((img_size, img_size, 3), base_color, dtype=np.uint8)
                noise = rng.integers(-15, 16,
                                      size=(img_size, img_size, 3)).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                # Add checkerboard artifact (simulates GAN upsampling artifact)
                checker = np.indices((img_size, img_size)).sum(axis=0) % 8 < 4
                img[checker, 0] = np.clip(
                    img[checker, 0].astype(int) + 20, 0, 255)
                cv2.imwrite(str(fake_dir / f"f{i:05d}_0000.jpg"), img)

    total = sum(splits.values()) * (1 + len(manips)) * 2
    print(f"[DEMO MODE] Generated ~{total} synthetic crops in '{output_dir}'.")
    print("  NOTE: These are purely synthetic — results will not be meaningful.")
    print("  Use this only to verify the pipeline runs end-to-end.\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    if args.demo_mode:
        generate_demo_data(args.output_dir, img_size=args.img_size, seed=args.seed)
        print("Done. Now run:")
        print("  python 03_train_efficientnet.py --demo_mode --epochs 2")
        return

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or derive video-ID splits
    # ------------------------------------------------------------------
    splits = load_splits(args.split_json)
    if splits is None:
        # Derive from real video filenames
        # real_video_dir = (data_root / "original_sequences" /
        #                   "youtube" / "c23" / "videos")
        real_video_dir = data_root / "original_sequences"
        if real_video_dir.exists():
            vid_ids = [p.stem for p in real_video_dir.glob("*.mp4")]
        else:
            vid_ids = [f"{i:03d}" for i in range(TOTAL_VIDEOS)]
        splits = assign_split_by_ratio(vid_ids, seed=args.seed)

    print(f"\nSplit sizes: "
          f"train={len(splits['train'])} | "
          f"val={len(splits['val'])} | "
          f"test={len(splits['test'])}")

    # ------------------------------------------------------------------
    # Initialise face detector
    # ------------------------------------------------------------------
    detector = FaceDetector()

    # ------------------------------------------------------------------
    # Extract real faces
    # ------------------------------------------------------------------
    # real_video_dir = (data_root / "original_sequences" /
    #                   "youtube" / "c23" / "videos")
    real_video_dir = data_root / "original_sequences"
    if real_video_dir.exists():
        print("\n[1/5] Extracting real faces...")
        n = extract_category(
            video_dir=real_video_dir,
            output_base=output_dir,
            category="real",
            splits=splits,
            detector=detector,
            frames_per_video=args.frames_per_video,
            img_size=args.img_size,
            margin=args.margin,
            seed=args.seed,
            max_videos=args.max_videos,
        )
        print(f"  Saved {n} real crops.")
    else:
        print(f"[WARN] Real video dir not found: {real_video_dir}")

    # ------------------------------------------------------------------
    # Extract fake faces for each manipulation type
    # ------------------------------------------------------------------
    for i, manip in enumerate(MANIPULATIONS, start=2):
        # fake_video_dir = (data_root / "manipulated_sequences" /
        #                   manip / "c23" / "videos")
        fake_video_dir = data_root / "manipulated_sequences" / manip
        if fake_video_dir.exists():
            print(f"\n[{i}/5] Extracting {manip} faces...")
            n = extract_category(
                video_dir=fake_video_dir,
                output_base=output_dir,
                category=manip,
                splits=splits,
                detector=detector,
                frames_per_video=args.frames_per_video,
                img_size=args.img_size,
                margin=args.margin,
                seed=args.seed,
                max_videos=args.max_videos,
            )
            print(f"  Saved {n} {manip} crops.")
        else:
            print(f"[WARN] Fake video dir not found: {fake_video_dir}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Preprocessing complete.")
    print(f"Output: {output_dir.resolve()}")
    for split in ["train", "val", "test"]:
        real_n = len(list((output_dir / split / "real").glob("*.jpg")))
        fake_n = sum(
            len(list((output_dir / split / "fake" / m).glob("*.jpg")))
            for m in MANIPULATIONS
            if (output_dir / split / "fake" / m).exists()
        )
        print(f"  {split:5s}: {real_n} real | {fake_n} fake")
    print("=" * 50)
    print("\nNext step:")
    print(f"  python 03_train_efficientnet.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
