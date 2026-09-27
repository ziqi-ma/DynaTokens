# Shared paths and defaults for the dynatokens scripts. Source, don't execute.
#
# Every variable can be overridden from the environment:
#   HYWP_ROOT     vendored HY-WorldPlay code         (default: <repo>/model/hywp)
#   OUTPUTS_ROOT  training data / runs / evals       (default: <repo>/outputs)
#   MODEL_PATH    HunyuanVideo-1.5 snapshot dir       (default: auto-detected in the HF cache)
#   ACTION_CKPT   HY-WorldPlay ar_rl_model safetensors (default: auto-detected in the HF cache)
#   CONDA_ENV     if set, `conda activate $CONDA_ENV` first; otherwise the current env is used

DYNATOKEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DYNATOKENS_DIR="$DYNATOKEN_ROOT/dynatokens"
: "${HYWP_ROOT:=$DYNATOKEN_ROOT/model/hywp}"
: "${OUTPUTS_ROOT:=$DYNATOKEN_ROOT/outputs}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="$(ls -d "$HF_HUB"/models--tencent--HunyuanVideo-1.5/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::' || true)"
fi
if [[ -z "${ACTION_CKPT:-}" ]]; then
    ACTION_CKPT="$(ls "$HF_HUB"/models--tencent--HY-WorldPlay/snapshots/*/ar_rl_model/diffusion_pytorch_model.safetensors 2>/dev/null | head -1 || true)"
fi
if [[ ! -d "$MODEL_PATH" || ! -f "$ACTION_CKPT" ]]; then
    echo "ERROR: HunyuanVideo-1.5 / HY-WorldPlay checkpoints not found." >&2
    echo "  MODEL_PATH=$MODEL_PATH" >&2
    echo "  ACTION_CKPT=$ACTION_CKPT" >&2
    echo "  Download them (see README 'Checkpoints') or set MODEL_PATH / ACTION_CKPT." >&2
    exit 1
fi

export DYNATOKEN_ROOT DYNATOKENS_DIR HYWP_ROOT OUTPUTS_ROOT MODEL_PATH ACTION_CKPT
export PYTHONPATH="$HYWP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
