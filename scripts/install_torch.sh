#!/usr/bin/env bash
set -euo pipefail

# Updated install_torch.sh
# - Uses nvcc / nvidia-smi to detect CUDA (you reported nvcc -> CUDA 11.5)
# - Maps CUDA 11.5 -> attempt cu118 wheel (no official cu115 wheel for torch 2.5.0)
# - Allows overriding wheel tag with TORCH_WHEEL_TAG env var
# - Uninstalls previous torch packages before installing
# - Tries fallback options on failure and prints helpful diagnostics

# Usage:
#  source .venv/bin/activate && ./scripts/install_torch.sh
#  Override wheel tag (if you know a better target): TORCH_WHEEL_TAG=cu118 ./scripts/install_torch.sh

TORCH_VERSION="${TORCH_VERSION:-2.5.0}"
# allow user to force wheel tag
FORCED_TAG="${TORCH_WHEEL_TAG:-}"

# detect CUDA via nvcc (preferred) or nvidia-smi (fallback)
CUDA_VER=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)
fi
if [ -z "$CUDA_VER" ] && command -v nvidia-smi >/dev/null 2>&1; then
  # nvidia-smi doesn't give toolkit version — use driver version as best-effort hint
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 || true)
fi

echo "Detected CUDA / driver (best-effort): '$CUDA_VER'"

# choose wheel tag (default mapping)
WHEEL_TAG="cpu"
if [ -n "$FORCED_TAG" ]; then
  echo "Using forced wheel tag from TORCH_WHEEL_TAG: $FORCED_TAG"
  WHEEL_TAG="$FORCED_TAG"
else
  case "$CUDA_VER" in
    12.*|12.1*)
      WHEEL_TAG="cu121"
      ;;
    11.8*)
      WHEEL_TAG="cu118"
      ;;
    11.7*)
      WHEEL_TAG="cu117"
      ;;
    11.6*)
      WHEEL_TAG="cu116"
      ;;
    11.5*)
      # NOTE: your nvcc shows 11.5; PyTorch 2.5.0 does not ship a +cu115 wheel.
      # Best practical choice is to try a newer runtime wheel (cu118) — the driver is usually compatible.
      echo "nvcc reports CUDA 11.5. There is no official torch+cu115 wheel for 2.5.0."
      echo "Attempting cu118 wheel (common & usually compatible); if that fails you can try cu121 or cpu."
      WHEEL_TAG="cu118"
      ;;
    11.*)
      # other 11.x -> try cu118 as a robust default for 11.x environments
      WHEEL_TAG="cu118"
      ;;
    *)
      # unknown -> fallback to cpu by default
      WHEEL_TAG="cpu"
      ;;
  esac
fi

echo "Selected wheel tag: $WHEEL_TAG"
echo "Torch version target: $TORCH_VERSION"

# ensure we run inside a venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: This script expects the venv to be activated first."
  echo "Run: source .venv/bin/activate"
  exit 2
fi

# uninstall any existing torch packages to avoid conflicts
echo "Uninstalling any existing torch / torchvision / torchaudio (if present)..."
pip uninstall -y torch torchvision torchaudio || true

# helper: perform install and return non-zero on failure
install_wheel() {
  local tag="$1"
  if [ "$tag" = "cpu" ]; then
    echo "Installing CPU-only wheels (torch==${TORCH_VERSION}+cpu)..."
    pip install --no-cache-dir "torch==${TORCH_VERSION}+cpu" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  else
    echo "Installing CUDA wheel index for tag: $tag"
    pip install --no-cache-dir "torch==${TORCH_VERSION}" torchvision torchaudio --index-url "https://download.pytorch.org/whl/${tag}"
  fi
}

# try the selected wheel, then fallback to cu118 (if not already), then cpu
TRY_ORDER=("$WHEEL_TAG")
if [ "$WHEEL_TAG" != "cu118" ]; then
  TRY_ORDER+=("cu118")
fi
if [ "${TRY_ORDER[-1]}" != "cpu" ]; then
  TRY_ORDER+=("cpu")
fi

INSTALL_SUCCESS=0
for tag in "${TRY_ORDER[@]}"; do
  echo "== Attempting install with tag: $tag =="
  set +e
  install_wheel "$tag"
  RC=$?
  set -e
  if [ $RC -eq 0 ]; then
    echo "Install with tag '$tag' succeeded."
    INSTALL_SUCCESS=1
    SELECTED_TAG="$tag"
    break
  else
    echo "[warn] Install with tag '$tag' failed (pip exit code $RC). Trying next fallback (if any)..."
  fi
done

if [ $INSTALL_SUCCESS -ne 1 ]; then
  echo "ERROR: All install attempts failed. Last tried tags: ${TRY_ORDER[*]}"
  echo "Advice:"
  echo " - Check your Python version (use a supported version like 3.9/3.10/3.11)."
  echo " - Manually pick a wheel tag from https://pytorch.org/get-started/previous-versions/ and retry with TORCH_WHEEL_TAG."
  echo " - Consider using conda if pip wheels are not available for your platform."
  exit 3
fi

# quick runtime check
echo "Running quick verification (import torch; print version info)..."
python - <<'PY'
import sys
try:
    import torch
    print("torch.__version__ =", torch.__version__)
    print("torch.version.cuda  =", getattr(torch.version, "cuda", None))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
except Exception as e:
    print("Error importing torch:", e, file=sys.stderr)
    sys.exit(1)
PY

# If cuda is not available but we installed a cuda wheel, print a helpful hint
if [ "${SELECTED_TAG:-}" != "cpu" ]; then
  python - <<'PY'
import torch,sys,subprocess
if not torch.cuda.is_available():
    print("\n[notice] torch was installed with CUDA support (tag=${SELECTED_TAG:-unknown}) but torch.cuda.is_available() is False.")
    print("Possible reasons:")
    print(" - NVIDIA driver incompatible/outdated for the CUDA runtime bundled in the wheel.")
    print(" - The GPU is not present/initialized or you're running inside a container without devices exposed.")
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print('\\nOutput of nvidia-smi:\\n' + (out.stdout or out.stderr))
    except Exception as e:
        print('nvidia-smi failed:', e)
    print("\\nRecommendation: update your NVIDIA driver to a recent version (or try a different wheel tag).")
    print("If problems persist consider using conda to install pytorch/pytorch-cuda which manages runtimes more robustly.")
    sys.exit(0)
else:
    print("\\nCUDA is available. Good to go!")
PY
fi

echo "Done. If you want to force a particular wheel tag next time use: TORCH_WHEEL_TAG=cu118 ./scripts/install_torch.sh"
