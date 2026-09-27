#!/bin/bash
# Stage 3: HYWP camera re-rendering (multi-GPU).
# GPU (HY-WorldPlay).
set -eo pipefail

: "${INSTANCES_JSON:?INSTANCES_JSON not set — source scripts_<bench>/config.sh + pipeline/defaults.sh first}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT not set}"
: "${POSES_JSON:?POSES_JSON not set (camera trajectories JSON, see README Camera Poses)}"
: "${CURATION_DIR:?CURATION_DIR not set}"

if [ -z "${MODEL_PATH:-}" ] || [ -z "${ACTION_CKPT:-}" ]; then
    echo "ERROR: MODEL_PATH or ACTION_CKPT not found."
    echo "  Download the checkpoints (see README 'Checkpoints') or set MODEL_PATH / ACTION_CKPT."
    exit 1
fi

activate_env "${HYWP_CONDA:-}"

echo "Step 3: HYWP camera re-rendering ($NUM_GPUS GPU(s))"

export PYTHONPATH="${HYWP_ROOT}:${PYTHONPATH:-}"
cd "$HYWP_ROOT"

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    cuda_dev=$((GPU_START + gpu_id))
    master_port=$((29500 + gpu_id))
    echo "  Starting GPU worker $gpu_id/$NUM_GPUS (CUDA_VISIBLE_DEVICES=$cuda_dev) ..."
    CUDA_VISIBLE_DEVICES=$cuda_dev MASTER_PORT=$master_port \
    python -u "$CURATION_DIR/hywp_rerender.py" \
        --instances_json "$INSTANCES_JSON" \
        --output_root  "$OUTPUT_ROOT" \
        --model_path   "$MODEL_PATH" \
        --action_ckpt  "$ACTION_CKPT" \
        --gpu_id       "$gpu_id" \
        --num_gpus     "$NUM_GPUS" \
        --poses_json   "$POSES_JSON" \
        --n_steps      "$N_STEPS" \
        --seed         "$SEED" \
        --num_inference_steps "$HYWP_STEPS" \
        ${MAX_INSTANCES:+--max_instances "$MAX_INSTANCES"} \
        ${MAX_POSES:+--max_poses "$MAX_POSES"} \
        "$@" \
        > >(sed "s/^/[GPU $gpu_id] /") 2>&1 &
    pids+=($!)
done

echo "  Waiting for $NUM_GPUS GPU worker(s) ..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: GPU worker PID $pid failed."
        failed=1
    fi
done

if [ $failed -ne 0 ]; then
    echo "Some GPU workers failed."
    exit 1
fi

echo "All GPU workers complete."
