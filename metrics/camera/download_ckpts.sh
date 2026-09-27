#!/usr/bin/env bash
# Trigger ViPE's lazy checkpoint downloads by running a single inference on
# the bundled demo video. Safe to re-run; checkpoints are cached under:
#
#   ~/.cache/huggingface/hub/            (Depth-Anything-V2, UniDepth, etc.)
#   $(python -c "import torch; print(torch.hub.get_dir())")/{sam,aot}/
#
# Prerequisite: the `vipe` conda env is active (run `conda activate vipe`
# or source setup.sh which does both).
#
# Note: the AOT tracker fetches its checkpoint via gdown from Google Drive,
# which is occasionally rate-limited. If it fails, wait a few minutes and
# re-run this script.

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIPE_DIR="$HERE/../../third_party/vipe"
DEMO_VIDEO="$VIPE_DIR/assets/examples/dog-example.mp4"
SMOKE_OUT="${VIPE_SMOKE_OUT:-/tmp/vipe_ckpt_warmup}"

if [ ! -f "$DEMO_VIDEO" ]; then
    echo "ERROR: demo video not found at $DEMO_VIDEO" >&2
    exit 1
fi

if ! command -v vipe >/dev/null 2>&1; then
    echo "ERROR: 'vipe' CLI not on PATH. Activate the conda env first:" >&2
    echo "    conda activate vipe" >&2
    exit 1
fi

echo "[download_ckpts.sh] Running smoke-test inference on $DEMO_VIDEO"
echo "[download_ckpts.sh] (Triggers checkpoint downloads; output -> $SMOKE_OUT)"
vipe infer "$DEMO_VIDEO" -o "$SMOKE_OUT"

echo "[download_ckpts.sh] Done. You can delete $SMOKE_OUT to reclaim space."
