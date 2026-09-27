#!/usr/bin/env bash
# End-to-end installer for the ViPE-based camera trajectory metric.
# Idempotent: re-running updates the env and retries ckpt downloads.
#
# Steps:
#   1. Create the `vipe` conda env (or $ENV_NAME) from environment.yml (or update it)
#   2. Install the pinned pip deps from vipe/envs/requirements.txt
#   3. Build + install vipe with its CUDA extensions (pip install -e .)
#   4. Prime checkpoint caches via a tiny smoke-test inference
#
# Usage:
#   git submodule update --init third_party/vipe
#   bash metrics/camera/setup.sh
#   ENV_NAME=<name> bash metrics/camera/setup.sh   to use another env name

# NOTE: -u is intentionally *off* here — NVIDIA's cuda-nvcc activate.d script
# references NVCC_PREPEND_FLAGS without a default, which trips `set -u`.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIPE_DIR="$HERE/../../third_party/vipe"
ENV_NAME="${ENV_NAME:-vipe}"

if [ ! -d "$VIPE_DIR" ]; then
    echo "ERROR: ViPE clone not found at $VIPE_DIR" >&2
    echo "Fetch it first: git submodule update --init third_party/vipe" >&2
    exit 1
fi

# ── 1. conda env ─────────────────────────────────────────────────────────────
echo "[setup.sh] Creating/updating conda env '$ENV_NAME' from $HERE/environment.yml"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$HERE/environment.yml"
else
    conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

# Activate it for the rest of the script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. pip deps ──────────────────────────────────────────────────────────────
echo "[setup.sh] Installing ViPE pinned pip deps (torch 2.7.0+cu128, ...)"
pip install -r "$VIPE_DIR/envs/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ── 3. build vipe with CUDA extensions ───────────────────────────────────────
echo "[setup.sh] Building vipe CUDA extensions (this takes a few minutes)"
# ViPE's setup.py honours CONDA_PREFIX/bin/nvcc automatically, but the host's
# /usr/bin/nvcc also matches CUDA 12.8 so the build works either way.
pip install --no-build-isolation -e "$VIPE_DIR"

# ── 4. prime checkpoint caches ───────────────────────────────────────────────
echo "[setup.sh] Priming checkpoint caches via a smoke-test inference"
bash "$HERE/download_ckpts.sh"

echo "[setup.sh] Done. Activate with: conda activate $ENV_NAME"
