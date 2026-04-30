# Deepfake Detection — Experimental Pipeline
## Securing Biometric Systems Against Deepfake Attacks

---

## Project Structure

```
deepfake_research/
├── README.md                    ← You are here
├── 00_setup_env.sh              ← Run FIRST: creates venv + installs deps
├── 01_download_dataset.sh       ← FF++ dataset download instructions
├── 02_preprocess.py             ← Face detection + crop extraction
├── 03_train_efficientnet.py     ← Method A: EfficientNet-B3 CNN trainer
├── 04_train_dct_classifier.py   ← Method B: DCT frequency-domain classifier
├── 05_evaluate.py               ← Cross-manipulation eval + metrics
├── 06_plot_results.py           ← Generate all figures for the paper
├── data/                        ← Put FF++ dataset here
│   └── FaceForensics++/
│       ├── original_sequences/
│       └── manipulated_sequences/
│           ├── Deepfakes/
│           ├── Face2Face/
│           ├── FaceSwap/
│           └── NeuralTextures/
├── models/                      ← Saved checkpoints
├── results/                     ← CSV metrics, confusion matrices
└── utils/
    ├── dataset.py               ← PyTorch Dataset class
    └── metrics.py               ← Evaluation helpers
```

---

## Step 0 — Prerequisites

- Python 3.10+ (venv built into Python, recommended)
- 8 GB RAM minimum (16 GB recommended)
- GPU optional but speeds up EfficientNet training significantly (even a cheap Colab GPU works)
- ~50 GB disk space for FF++ C23

---

## Step 1 — Environment Setup

<!-- ### Option A — Automated

Run the helper script to create a virtual environment and install the dependencies from `requirements.txt`:

```bash
bash 00_setup_env.sh -->
```


```bash
python -m venv deepfake_env
# or
py -3.12 -m venv deepfake_env # to avoid version issues
source deepfake_env/bin/activate   # Linux / Mac
# or
deepfake_env\Scripts\activate     # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
#or
pip install --only-binary :all: -r requirements.txt
```

If you have a CUDA-capable GPU and want GPU support, install the appropriate PyTorch wheel after activating the virtual environment instead of the default CPU wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

```

After activation, run the rest of the pipeline commands from the same shell.

---

## Step 2 — Dataset Download

FaceForensics++ requires a one-time access request form:

1. Fill the form at: https://github.com/ondyari/FaceForensics
   (Takes ~24h to get the download script) or download dataset from kaggle : https://www.kaggle.com/datasets/xdxd003/ff-c23?resource=download

2. Once you have `download-FaceForensics.py`, run:

```bash
# Download ONLY the C23 (light compression) version to save space
# Downloads ~30 GB total for all 4 manipulation types + originals

python download-FaceForensics.py 
    data/FaceForensics++ 
    -d all 
    -c c23 
    -t videos 
    --server EU

# 'all' downloads: original + Deepfakes + Face2Face + FaceSwap + NeuralTextures
```

3. After download, run preprocessing:

```bash
python 02_preprocess.py --data_root data/FaceForensics++_C23 --output_dir data/faces
```
---

## Step 3 — Run Method A (EfficientNet-B3)

```bash
# Train on all manipulation types (binary: real vs fake)
python 03_train_efficientnet.py 
    --data_dir data/faces 
    --manip all 
    --epochs 10 
    --batch_size 32 
    --output_dir models/efficientnet

# Train on a single manipulation type (for cross-manipulation experiment)
python 03_train_efficientnet.py --data_dir data/faces --manip Deepfakes
python 03_train_efficientnet.py --data_dir data/faces --manip Face2Face
python 03_train_efficientnet.py --data_dir data/faces --manip FaceSwap
python 03_train_efficientnet.py --data_dir data/faces --manip NeuralTextures
```

---

## Step 4 — Run Method B (DCT Classifier)

```bash
python 04_train_dct_classifier.py 
    --data_dir data/faces 
    --manip all 
    --output_dir models/dct

# Per-manipulation for cross-manip experiment
python 04_train_dct_classifier.py --data_dir data/faces --manip Deepfakes
python 04_train_dct_classifier.py --data_dir data/faces --manip Face2Face
python 04_train_dct_classifier.py --data_dir data/faces --manip FaceSwap
python 04_train_dct_classifier.py --data_dir data/faces --manip NeuralTextures
```

---

## Step 5 — Full Evaluation

```bash
python 05_evaluate.py 
    --efficientnet_dir models/efficientnet 
    --dct_dir models/dct 
    --data_dir data/faces 
    --output_dir results/
```

This generates:
- `results/main_results.csv`      — Table 2 in paper
- `results/cross_manip.csv`       — Table 3 in paper  
- `results/confusion_matrices/`   — Figures in paper

---

## Step 6 — Generate Paper Figures

```bash
python 06_plot_results.py --results_dir results/ --output_dir results/figures/
```

---

## Quick Test (No GPU, Small Data)

If you want to test the pipeline quickly without the full dataset:

```bash
python 02_preprocess.py --demo_mode   # generates synthetic demo faces
python 03_train_efficientnet.py --demo_mode --epochs 2
python 04_train_dct_classifier.py --demo_mode
python 05_evaluate.py --demo_mode
```

