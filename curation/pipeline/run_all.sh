#!/bin/bash
# Orchestrate all curation pipeline stages in order.
#
# Requires env (from scripts_<bench>/config.sh + pipeline/defaults.sh):
#   INSTANCES_JSON, OUTPUT_ROOT, CURATION_DIR, PIPELINE_DIR (+ per-stage vars)
set -eo pipefail

: "${PIPELINE_DIR:?PIPELINE_DIR not set — source scripts_<bench>/config.sh first}"
: "${POSES_JSON:?POSES_JSON not set (camera trajectories JSON, see README Camera Poses)}"

# Validate the poses file up front, before any API calls.
python -c "import sys; sys.path.insert(0, sys.argv[1]); from keyframe_utils import load_poses; \
p, n = load_poses(sys.argv[2], n_steps=int(sys.argv[3])); print(f'{len(p)} camera poses from {sys.argv[2]}')" \
    "$CURATION_DIR" "$POSES_JSON" "$N_STEPS"

bash "$PIPELINE_DIR/generate_states.sh"
echo ""
bash "$PIPELINE_DIR/deblur.sh"
echo ""
bash "$PIPELINE_DIR/hywp_rerender.sh"
echo ""
bash "$PIPELINE_DIR/assemble.sh"

echo ""
echo "Pipeline complete!"
