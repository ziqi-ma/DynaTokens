#!/bin/bash
# Stages 4-5: Assemble keyframes, Kling interpolation, stitch.
# API only (Gemini / Kling / Veo).
set -eo pipefail

: "${INSTANCES_JSON:?INSTANCES_JSON not set — source scripts_<bench>/config.sh + pipeline/defaults.sh first}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${POSES_JSON:?POSES_JSON not set (camera trajectories JSON, see README Camera Poses)}"
: "${CURATION_DIR:?CURATION_DIR not set}"

activate_env "${API_CONDA:-}"
require_api_keys GOOGLE_API_KEY FAL_KEY

echo "Steps 4-5: Assemble, interpolate, stitch"

cd "$CURATION_DIR"
python -u assemble_keyframes.py \
    --instances_json "$INSTANCES_JSON" \
    --output_root  "$OUTPUT_ROOT" \
    --poses_json   "$POSES_JSON" \
    --n_steps      "$N_STEPS" \
    --workers      "$WORKERS" \
    ${MAX_INSTANCES:+--max_instances "$MAX_INSTANCES"} \
    ${MAX_POSES:+--max_poses "$MAX_POSES"} \
    "$@"
