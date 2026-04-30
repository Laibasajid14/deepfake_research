#!/usr/bin/env bash
# =============================================================================
# 00_setup_env.sh
# Creates a fresh Python virtual environment and installs dependencies.
# Usage: bash 00_setup_env.sh
# =============================================================================

set -e   # exit on first error

ENV_DIR="deepfake_env"

echo "=============================================="
echo "  Deepfake Detection — venv Setup"
echo "=============================================="

# --------------------------------------------------------------------------
# 1. Find an available Python 3 interpreter
# --------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python 3 is required but not found."
    echo "  Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

if ! ${PYTHON_CMD} -c 'import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)'
then
    FOUND_VERSION=$(${PYTHON_CMD} -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')
    echo "[ERROR] Python 3.10+ is required. Found Python ${FOUND_VERSION}."
    exit 1
fi

# --------------------------------------------------------------------------
# 2. Create fresh venv
# --------------------------------------------------------------------------
if [ -d "${ENV_DIR}" ]; then
    echo "[INFO] Removing existing virtual environment '${ENV_DIR}'..."
    rm -rf "${ENV_DIR}"
fi

echo "[INFO] Creating virtual environment '${ENV_DIR}'..."
${PYTHON_CMD} -m venv "${ENV_DIR}"

# --------------------------------------------------------------------------
# 3. Activate and install packages
# --------------------------------------------------------------------------
if [ -f "${ENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ENV_DIR}/bin/activate"
elif [ -f "${ENV_DIR}/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${ENV_DIR}/Scripts/activate"
else
    echo "[WARN] Could not auto-activate the virtual environment."
    echo "  Activate it manually after the script finishes."
fi

echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

echo "[INFO] Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# 4. Verify installation
# --------------------------------------------------------------------------
echo ""
echo "[INFO] Verifying installation..."
python - <<'PYCHECK'
import torch, torchvision, sklearn, cv2, numpy, matplotlib, tqdm, timm, facenet_pytorch
print(f"  torch        : {torch.__version__}")
print(f"  torchvision  : {torchvision.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  opencv       : {cv2.__version__}")
print(f"  numpy        : {numpy.__version__}")
print(f"  timm         : {timm.__version__}")
print(f"  facenet-pytorch: {facenet_pytorch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
PYCHECK

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "  Activate with:"
echo "    source ${ENV_DIR}/bin/activate   # Linux / Mac"
echo "    ${ENV_DIR}\\Scripts\\activate     # Windows PowerShell"
echo "=============================================="
