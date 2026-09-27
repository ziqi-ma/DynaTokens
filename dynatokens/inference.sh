#!/bin/bash
# Render camera trajectories with a trained DynaToken checkpoint (AR HY-WorldPlay, 480p i2v).
#
# Released checkpoint (folder with diffusion_pytorch_model.safetensors, init_image.png,
# prompt.txt, sample_poses/<name>/pose.json): renders every sample pose into OUT_DIR
# [outputs/demo/<folder name>/<name>/gen.mp4]:
#   CKPT=path/to/vbench_sweep bash dynatokens/inference.sh
#
# Otherwise:
#   CKPT        <run>/checkpoint-N, its transformer/diffusion_pytorch_model.safetensors,
#               a released checkpoint folder, or "none" to render with the base model
#   and one of
#   DATA_BASE   a prepared scene (prepare_training_data.py): renders its held-out test/
#               trajectories (and, with SEEN=1, its training trajectories) into
#               OUT_DIR [<run>/eval/ckpt<N>, or outputs/eval/base/<scene> for CKPT=none]
#               as {unseen,seen}_<name>/gen.mp4
#   POSE_JSON + IMAGE_PATH + OUT_DIR   a single trajectory (e.g. from make_pose_json.py);
#               IMAGE_PATH is the first-frame image, or a GT video whose first frame is used
#               (then also scored with frame MSE)
#   JOBS_JSON   [{"pose_json": ..., "image_path": ..., "output_dir": ..., "prompt": ""}, ...]
#
# Optional: PROMPT, NAMES (subset of trajectory names, DATA_BASE mode), NUM_GPUS [4],
#           CUDA_VISIBLE_DEVICES [0,1,2,3], STEPS [30], SEED [42],
#           GROUP_OFFLOADING=1 to reduce GPU memory
#
# Output: <output_dir>/gen.mp4 (+ metrics.json when a GT video is given)
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${CKPT:?set CKPT to a checkpoint (or none)}"
NUM_GPUS=${NUM_GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

ckpt_arg=()
if [[ "$CKPT" != "none" ]]; then
    if [[ -d "$CKPT" && -f "$CKPT/diffusion_pytorch_model.safetensors" ]]; then
        RELEASED_DIR="$(cd "$CKPT" && pwd)"                                   # released checkpoint
        CKPT="$RELEASED_DIR/diffusion_pytorch_model.safetensors"
    elif [[ -d "$CKPT" ]]; then
        CKPT="$CKPT/transformer/diffusion_pytorch_model.safetensors"         # training checkpoint
    fi
    [[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
    ckpt_arg=(--temporal_embed_ckpt "$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")")
fi

if [[ -n "${RELEASED_DIR:-}" && -z "${DATA_BASE:-}${POSE_JSON:-}${JOBS_JSON:-}" ]]; then
    OUT_DIR=${OUT_DIR:-$OUTPUTS_ROOT/demo/$(basename "$RELEASED_DIR")}
    mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"
    JOBS_JSON="$OUT_DIR/jobs.json"
    python - "$RELEASED_DIR" "$OUT_DIR" "$JOBS_JSON" <<'PYEOF'
import glob, json, os, sys
ckpt_dir, out_dir, jobs_json = sys.argv[1:]
prompt_file = os.path.join(ckpt_dir, "prompt.txt")
prompt = open(prompt_file).read().strip() if os.path.isfile(prompt_file) else ""
def init_image(pose_json):
    # A pose rendered from a mid-trajectory state ships its own init_image.png
    # next to its pose.json; the checkpoint-level image covers the rest.
    per_pose = os.path.join(os.path.dirname(pose_json), "init_image.png")
    return per_pose if os.path.isfile(per_pose) else os.path.join(ckpt_dir, "init_image.png")
jobs = [{"pose_json": p, "image_path": init_image(p),
         "output_dir": os.path.join(out_dir, os.path.basename(os.path.dirname(p))), "prompt": prompt}
        for p in sorted(glob.glob(os.path.join(ckpt_dir, "sample_poses", "*", "pose.json")))]
json.dump(jobs, open(jobs_json, "w"), indent=2)
# scene.json next to the renders, as make_inference_jobs.py does (read by the metrics)
json.dump({"prompt": prompt}, open(os.path.join(out_dir, "scene.json"), "w"))
print(f"{len(jobs)} sample poses -> {out_dir}")
PYEOF
fi

if [[ -n "${DATA_BASE:-}" ]]; then
    if [[ -z "${OUT_DIR:-}" ]]; then
        if [[ "$CKPT" == "none" ]]; then
            OUT_DIR="$OUTPUTS_ROOT/eval/base/$(basename "$DATA_BASE")"
        else
            ckpt_dir="$(dirname "$(dirname "$(realpath "$CKPT")")")"          # <run>/checkpoint-N
            OUT_DIR="$(dirname "$ckpt_dir")/eval/ckpt${ckpt_dir##*checkpoint-}"
        fi
    fi
    mkdir -p "$OUT_DIR"
    JOBS_JSON="$OUT_DIR/jobs.json"
    python "$DYNATOKENS_DIR/make_inference_jobs.py" --data-base "$DATA_BASE" --out-dir "$OUT_DIR" \
        --jobs-json "$JOBS_JSON" --prompt "${PROMPT:-}" ${SEEN:+--seen} ${NAMES:+--names $NAMES}
    if [[ "$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$JOBS_JSON")" == 0 ]]; then
        echo "Nothing to render in $OUT_DIR"; exit 0
    fi
fi

if [[ -n "${JOBS_JSON:-}" ]]; then
    job_args=(--jobs_json "$(cd "$(dirname "$JOBS_JSON")" && pwd)/$(basename "$JOBS_JSON")")
else
    : "${POSE_JSON:?set JOBS_JSON, or POSE_JSON + IMAGE_PATH + OUT_DIR}"
    : "${IMAGE_PATH:?set IMAGE_PATH}"
    : "${OUT_DIR:?set OUT_DIR}"
    mkdir -p "$OUT_DIR"
    job_args=(--pose_json "$(realpath "$POSE_JSON")" --image_path "$(realpath "$IMAGE_PATH")"
              --output_dir "$(realpath "$OUT_DIR")" --prompt "${PROMPT:-}")
fi

cd "$HYWP_ROOT"
torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-$((29500 + RANDOM % 1000))}" \
    scripts/inference.py \
    --model_path "$MODEL_PATH" \
    --action_ckpt "$ACTION_CKPT" \
    "${ckpt_arg[@]}" \
    "${job_args[@]}" \
    --num_inference_steps "${STEPS:-30}" \
    --seed "${SEED:-42}" \
    ${GROUP_OFFLOADING:+--enable_group_offloading}
