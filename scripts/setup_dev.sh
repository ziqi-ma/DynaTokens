#!/usr/bin/env bash
# Create or update the main DynaTokens development environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-dynatoken}"
INSTALL_FLASH_ATTN=0
DOWNLOAD_MODELS=0
HF_TOKEN_VALUE="${HF_TOKEN:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_dev.sh [options]

Options:
  --flash-attn       Install the optional FlashAttention package.
  --download-models  Download HY-WorldPlay and FLUX.1-Redux checkpoints.
  --hf-token TOKEN   Hugging Face token used with --download-models.
  -h, --help         Show this help message.

Environment:
  ENV_NAME           Conda environment name (default: dynatoken).
  HF_TOKEN           Hugging Face token, equivalent to --hf-token.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --flash-attn)
            INSTALL_FLASH_ATTN=1
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=1
            shift
            ;;
        --hf-token)
            [[ $# -ge 2 ]] || { echo "ERROR: --hf-token requires a value" >&2; exit 2; }
            HF_TOKEN_VALUE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda was not found. Install Miniconda or Anaconda first." >&2
    exit 1
}

# Make conda activate available in this non-interactive shell.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Updating Conda environment '$ENV_NAME'"
    conda env update --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml" --prune
else
    echo "Creating Conda environment '$ENV_NAME'"
    conda env create --name "$ENV_NAME" --file "$ROOT_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

if [[ "$INSTALL_FLASH_ATTN" == 1 ]]; then
    echo "Installing optional FlashAttention"
    python -m pip install flash-attn --no-build-isolation
fi

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
    [[ -n "$HF_TOKEN_VALUE" ]] || {
        echo "ERROR: --download-models requires --hf-token or HF_TOKEN" >&2
        exit 2
    }
    echo "Downloading HY-WorldPlay and FLUX.1-Redux checkpoints"
    PYTHONPATH="$ROOT_DIR/model/hywp${PYTHONPATH:+:$PYTHONPATH}" \
        python "$ROOT_DIR/model/hywp/download_models.py" --hf_token "$HF_TOKEN_VALUE"
fi

echo "Developer environment is ready. Activate it with: conda activate $ENV_NAME"
