#!/bin/bash
# Curate supervision videos for a set of scenes (any dataset).
#
#   INSTANCES_JSON  (required) [{"image": ..., "prompt": ..., "output": ...}, ...]
#                   (see README "Instance JSON Format"; relative image paths resolve against DATA_ROOT;
#                   convert_worldscore.py / convert_physicsiq.py build it from those benchmarks)
#   OUTPUT_ROOT     (required) output root; each scene goes to $OUTPUT_ROOT/<output>
#   POSES_JSON      (required) camera trajectories (see README "Camera Poses")
#   N_STEPS         segments per pose [3]
#   plus any pipeline/defaults.sh knob
#
# Example:
#   INSTANCES_JSON=my_scenes.json POSES_JSON=my_poses.json OUTPUT_ROOT=outputs/curation/my_scenes \
#       N_STEPS=2 bash curation/run.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${INSTANCES_JSON:?set INSTANCES_JSON}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
INSTANCES_JSON="$(cd "$(dirname "$INSTANCES_JSON")" && pwd)/$(basename "$INSTANCES_JSON")"
mkdir -p "$OUTPUT_ROOT"; OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export INSTANCES_JSON OUTPUT_ROOT
source "$SCRIPT_DIR/pipeline/defaults.sh"

bash "$PIPELINE_DIR/run_all.sh"
