#!/usr/bin/env bash
# ============================================================================
# MiniMax-H3 headless runner launcher (Linux / GPU)
# ----------------------------------------------------------------------------
# Bootstraps the vendored comfy_core package and runs inference/run_minimax_h3.py
# (no ComfyUI server). All args are forwarded to the Python runner.
#
# Usage examples:
#   ./inference/run.sh --check
#   ./inference/run.sh --prompt "a cat waving" --mode reference \
#       --ref-image-1 ref.png --output out.mp4
#   ./inference/run.sh --config job.json
#   ./inference/run.sh --config job.json --pass2-config pass2.json
#
# Optional env:
#   MODELS_DIR    weights root (default: <repo>/models)
#   PYTHON        interpreter  (default: python3)
#   PIP_INSTALL   set to 1 to `pip install -r requirements-linux.txt` first
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_CORE="$REPO_ROOT/comfy_core"
INFERENCE_DIR="$REPO_ROOT/inference"

export PYTHONPATH="$COMFY_CORE:${PYTHONPATH:-}"
export MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"

PYTHON="${PYTHON:-python3}"

echo "[run.sh] REPO_ROOT   = $REPO_ROOT"
echo "[run.sh] COMFY_CORE  = $COMFY_CORE"
echo "[run.sh] MODELS_DIR  = $MODELS_DIR"

if [[ "${PIP_INSTALL:-0}" == "1" ]]; then
  echo "[run.sh] installing python deps from requirements-linux.txt ..."
  "$PYTHON" -m pip install -r "$INFERENCE_DIR/requirements-linux.txt"
fi

# Make sure the weight sub-folders exist (folder_paths shim reads these).
mkdir -p "$MODELS_DIR"/{unet,vae,clip,loras}

exec "$PYTHON" "$INFERENCE_DIR/run_minimax_h3.py" "$@"
