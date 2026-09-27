#!/bin/bash
# Stage 1: Generate dynamics video (Kling/Veo) + extract state keyframes.
# API only (Gemini / Kling / Veo).
#
# Requires env (from scripts_<bench>/config.sh + pipeline/defaults.sh):
#   INSTANCES_JSON, DATA_ROOT, OUTPUT_ROOT, VIDEO_MODEL, DURATION, N_STEPS,
#   GOOGLE_API_KEY, FAL_KEY, CURATION_DIR
# Optional: MAX_INSTANCES
set -eo pipefail

: "${INSTANCES_JSON:?INSTANCES_JSON not set — source scripts_<bench>/config.sh + pipeline/defaults.sh first}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${CURATION_DIR:?CURATION_DIR not set}"

activate_env "${API_CONDA:-}"
require_api_keys GOOGLE_API_KEY FAL_KEY

echo "Step 1: Generate dynamics video ($VIDEO_MODEL) + extract keyframes"

cd "$CURATION_DIR"
python -u generate_states.py \
    --instances_json "$INSTANCES_JSON" \
    --data_root    "$DATA_ROOT" \
    --output_root  "$OUTPUT_ROOT" \
    --video_model  "$VIDEO_MODEL" \
    --duration     "$DURATION" \
    --n_steps      "$N_STEPS" \
    ${MAX_INSTANCES:+--max_instances "$MAX_INSTANCES"} \
    "$@"
