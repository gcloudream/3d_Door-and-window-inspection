#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINT_SAM_REPO_DIR="${POINT_SAM_REPO_DIR:-$PROJECT_ROOT/.vendor/Point-SAM}"
POINT_SAM_VENV_DIR="${POINT_SAM_VENV_DIR:-$PROJECT_ROOT/.venv-point-sam}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/7] Checking required commands"
command -v "$PYTHON_BIN" >/dev/null
command -v git >/dev/null

if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found. This setup script is intended for a Linux GPU machine." >&2
  exit 1
fi

echo "[2/7] Preparing directories"
mkdir -p "$(dirname "$POINT_SAM_REPO_DIR")"

echo "[3/7] Creating virtual environment at $POINT_SAM_VENV_DIR"
"$PYTHON_BIN" -m venv "$POINT_SAM_VENV_DIR"
# shellcheck disable=SC1090
source "$POINT_SAM_VENV_DIR/bin/activate"

echo "[4/7] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[5/7] Verifying PyTorch is installed inside the virtualenv"
if ! python - <<'PY'
import importlib.util
import sys
spec = importlib.util.find_spec("torch")
sys.exit(0 if spec is not None else 1)
PY
then
  cat >&2 <<'EOF'
ERROR: torch is not installed in the selected environment.
Install GPU-enabled PyTorch first using the official selector:
https://pytorch.org/get-started/locally/

Then rerun this script.
EOF
  exit 1
fi

echo "[6/7] Cloning or updating Point-SAM"
if [ ! -d "$POINT_SAM_REPO_DIR/.git" ]; then
  git clone https://github.com/zyc00/Point-SAM.git "$POINT_SAM_REPO_DIR"
else
  git -C "$POINT_SAM_REPO_DIR" pull --ff-only
fi

echo "[7/7] Installing Point-SAM Python dependencies"
python -m pip install "timm>=0.9.0" "torchvision>=0.16.0" hydra-core omegaconf accelerate safetensors numpy flask flask-cors

git -C "$POINT_SAM_REPO_DIR" submodule update --init third_party/torkit3d third_party/apex

if command -v conda >/dev/null; then
  echo "Installing the compiler recommended by the official README through conda"
  conda install -y gxx_linux-64=9.3.0 || true
fi

echo "Installing third_party/torkit3d"
FORCE_CUDA=1 python -m pip install "$POINT_SAM_REPO_DIR/third_party/torkit3d"

echo "Installing third_party/apex"
python -m pip install \
  -v \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  "$POINT_SAM_REPO_DIR/third_party/apex"

cat <<EOF

Point-SAM base environment is ready.

Next steps:
1. Download the official checkpoint to a local path, for example:
   $POINT_SAM_REPO_DIR/pretrained/model.safetensors
2. Export runtime variables:
   export POINT_SEGMENTER_BACKEND=point_sam
   export POINT_SAM_REPO_DIR=$POINT_SAM_REPO_DIR
   export POINT_SAM_CHECKPOINT=/absolute/path/to/model.safetensors
   export POINT_SAM_DEVICE=cuda
3. Validate runtime:
   $POINT_SAM_VENV_DIR/bin/python $PROJECT_ROOT/scripts/point_sam_doctor.py
4. Run the demo:
   POINT_SAM_VENV_DIR=$POINT_SAM_VENV_DIR POINT_SAM_REPO_DIR=$POINT_SAM_REPO_DIR $PROJECT_ROOT/scripts/run_demo_point_sam.sh

EOF
