#!/usr/bin/env bash
# =============================================================================
# 01_download_dataset.sh
# Helper script for downloading FaceForensics++ (C23 compression).
#
# IMPORTANT: FF++ requires a one-time access request.
# Fill the form at:  https://github.com/ondyari/FaceForensics
# You will receive the official download script by email (usually within 24h).
# =============================================================================

set -e

DATA_ROOT="data/FaceForensics++"
DOWNLOAD_SCRIPT="download-FaceForensics.py"   # provided by the FF++ authors

echo "=============================================="
echo "  FaceForensics++ Dataset Download Helper"
echo "=============================================="

# --------------------------------------------------------------------------
# Step 1: Check the download script exists
# --------------------------------------------------------------------------
if [ ! -f "${DOWNLOAD_SCRIPT}" ]; then
    echo ""
    echo "[ERROR] '${DOWNLOAD_SCRIPT}' not found in the current directory."
    echo ""
    echo "To get it:"
    echo "  1. Open https://github.com/ondyari/FaceForensics"
    echo "  2. Scroll to 'Access' and fill the Google Form."
    echo "  3. You will receive an email with a link to download-FaceForensics.py"
    echo "  4. Place that file in this directory and re-run this script."
    echo ""
    exit 1
fi

mkdir -p "${DATA_ROOT}"

echo ""
echo "[INFO] Downloading FF++ C23 videos (~30 GB total)..."
echo "[INFO] Using EU server — change '--server EU' to 'EU2' or 'CA' if slow."
echo ""

# --------------------------------------------------------------------------
# Download original (real) videos
# --------------------------------------------------------------------------
echo "--- Downloading original sequences ---"
python "${DOWNLOAD_SCRIPT}" \
    "${DATA_ROOT}" \
    -d original \
    -c c23 \
    -t videos \
    --server EU

# --------------------------------------------------------------------------
# Download each manipulation type
# --------------------------------------------------------------------------
for MANIP in Deepfakes Face2Face FaceSwap NeuralTextures; do
    echo ""
    echo "--- Downloading ${MANIP} ---"
    python "${DOWNLOAD_SCRIPT}" \
        "${DATA_ROOT}" \
        -d "${MANIP}" \
        -c c23 \
        -t videos \
        --server EU
done

# --------------------------------------------------------------------------
# Download official train/val/test splits JSON
# --------------------------------------------------------------------------
echo ""
echo "--- Downloading dataset splits ---"
python "${DOWNLOAD_SCRIPT}" \
    "${DATA_ROOT}" \
    -d original \
    -c c23 \
    -t json \
    --server EU

echo ""
echo "=============================================="
echo "  Download complete."
echo ""
echo "  Expected directory structure:"
echo "    ${DATA_ROOT}/"
echo "    ├── original_sequences/"
echo "    │   └── youtube/c23/videos/"
echo "    └── manipulated_sequences/"
echo "        ├── Deepfakes/c23/videos/"
echo "        ├── Face2Face/c23/videos/"
echo "        ├── FaceSwap/c23/videos/"
echo "        └── NeuralTextures/c23/videos/"
echo ""
echo "  Next step:"
echo "    python 02_preprocess.py --data_root ${DATA_ROOT} --output_dir data/faces"
echo "=============================================="
