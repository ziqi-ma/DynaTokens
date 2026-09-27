#!/bin/bash
# Shared defaults for the curation (dynamics data engine) pipeline.
#
# Sourced by scripts_<benchmark>/config.sh after benchmark-specific paths are
# set. Uses `: "${VAR:=default}"` so any value pre-set by the caller
# (environment or benchmark config.sh) wins.

# ── Directory anchors ───────────────────────────────────────────────────────
: "${CURATION_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${PIPELINE_DIR:=$CURATION_DIR/pipeline}"
: "${DYNATOKEN_ROOT:=$(cd "$CURATION_DIR/.." && pwd)}"
: "${DATA_ROOT:=$DYNATOKEN_ROOT/datasets}"
: "${HYWP_ROOT:=$DYNATOKEN_ROOT/model/hywp}"

# ── Pipeline settings ───────────────────────────────────────────────────────
: "${VIDEO_MODEL:=kling}"   # veo or kling
: "${NUM_GPUS:=4}"
: "${GPU_START:=0}"
: "${DURATION:=5}"
: "${N_STEPS:=3}"           # segments per camera pose (= state keyframes); must match POSES_JSON
: "${WORKERS:=8}"
: "${SEED:=42}"
: "${HYWP_STEPS:=30}"
: "${MAX_POSES:=}"
: "${MAX_INSTANCES:=}"

# ── Camera trajectories ─────────────────────────────────────────────────────
# POSES_JSON (required by stages 3-5): {"train": {"<name>": "<pose>"}, "test": {...}}
# Every pose string must have N_STEPS segments, e.g. "w-6, right-7, w-6" for N_STEPS=3.
if [ -n "${POSES_JSON:-}" ]; then POSES_JSON="$(cd "$(dirname "$POSES_JSON")" && pwd)/$(basename "$POSES_JSON")"; fi

# ── API keys (required by the API stages; never hard-code them) ────────────
#   GOOGLE_API_KEY  Gemini (motion planning, quality check, prompts) + Veo
#   FAL_KEY         Kling via fal.ai (video generation + interpolation)
require_api_keys() {
    local missing=0
    for k in "$@"; do
        if [ -z "${!k:-}" ]; then echo "ERROR: $k is not set (export it before running)." >&2; missing=1; fi
    done
    [ $missing -eq 0 ] || exit 1
}

# ── Environments ────────────────────────────────────────────────────────────
# By default every stage runs in the current Python environment. To use
# separate conda envs for the API stages and the HYWP (GPU) stage, set
# API_CONDA and/or HYWP_CONDA to their names.
activate_env() {
    if [ -n "${1:-}" ]; then
        eval "$(conda shell.bash hook)"
        conda activate "$1"
    fi
}

# ── HYWP checkpoint auto-detection (from HF cache) ──────────────────────────
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(ls -d "$HF_HUB"/models--tencent--HunyuanVideo-1.5/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::' || true)"
fi
if [ -z "${ACTION_CKPT:-}" ]; then
    ACTION_CKPT="$(ls "$HF_HUB"/models--tencent--HY-WorldPlay/snapshots/*/ar_rl_model/diffusion_pytorch_model.safetensors 2>/dev/null | head -1 || true)"
fi

# ── Export for child processes ──────────────────────────────────────────────
export CURATION_DIR PIPELINE_DIR DYNATOKEN_ROOT DATA_ROOT HYWP_ROOT
export VIDEO_MODEL NUM_GPUS GPU_START DURATION N_STEPS WORKERS SEED HYWP_STEPS
export MAX_POSES MAX_INSTANCES POSES_JSON
export API_CONDA HYWP_CONDA
export MODEL_PATH ACTION_CKPT
export -f require_api_keys activate_env
