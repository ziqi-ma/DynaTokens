#!/bin/bash
# Stage 2: De-blur state keyframes.
# API only (Gemini / Kling / Veo).
set -eo pipefail

: "${INSTANCES_JSON:?INSTANCES_JSON not set — source scripts_<bench>/config.sh + pipeline/defaults.sh first}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${CURATION_DIR:?CURATION_DIR not set}"

activate_env "${API_CONDA:-}"
require_api_keys GOOGLE_API_KEY

echo "Deblur: state images for all instances"

cd "$CURATION_DIR"
python -u - <<'EOF'
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

instances_json = os.environ["INSTANCES_JSON"]
output_root    = os.environ["OUTPUT_ROOT"]
n_steps        = int(os.environ.get("N_STEPS", "3"))
deblur_script  = os.path.join(os.getcwd(), "deblur_keyframes.py")

with open(instances_json) as f:
    instances = json.load(f)

def deblur_inst(inst):
    output = inst["output"]
    inst_dir = os.path.join(output_root, output)
    states_dir = os.path.join(inst_dir, "states")
    if not os.path.isdir(states_dir):
        print(f"  [{output}] No states dir, skipping.")
        return
    print(f"  [{output}] De-blurring ...")
    subprocess.run(
        [sys.executable, deblur_script,
         "--states-dir", states_dir,
         "--n-steps", str(n_steps)],
        check=False,
    )

with ThreadPoolExecutor(max_workers=len(instances)) as pool:
    futs = [pool.submit(deblur_inst, inst) for inst in instances]
    for fut in as_completed(futs):
        fut.result()

print("Deblur complete.")
EOF
