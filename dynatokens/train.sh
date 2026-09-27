#!/bin/bash
# Train DynaToken (per-block temporal cross-attention) for one scene.
#
# The base model is frozen; only the temporal cross-attention
# branch (learned tokens + K/V projections + zero-initialised out-projection,
# one per double-stream block) is trained.
#
# Required:
#   DATA_BASE        prepared scene dir (output of prepare_training_data.py)
#
# Optional (defaults in brackets):
#   WINDOW_FRAMES [from $DATA_BASE/scene.json]  latent frames per training video
#   RUN_NAME [<basename DATA_BASE>_<hash>]   OUTPUT_DIR [$OUTPUTS_ROOT/runs/$RUN_NAME]
#   DATA_JSON_PATH [$DATA_BASE/train_all_textprompt.json]
#   LEARNING_RATE [3e-3]  WEIGHT_DECAY [1e-5]  CROSSATTN_TOKEN_DIM [64]
#   MAX_TRAIN_STEPS [3000]  CHECKPOINTING_STEPS [500]  VALIDATION_STEPS [= CHECKPOINTING_STEPS]
#   NUM_GPUS [4]  CUDA_VISIBLE_DEVICES [0,1,2,3]  MASTER_PORT [random]
#   RESUME_FROM_CHECKPOINT  path to a previous <run>/checkpoint-N
#   In-training video eval, rendered every VALIDATION_STEPS from the checkpoint just saved
#   into $OUTPUT_DIR/eval_step<N>_{seen,unseen}/:
#     EVAL_POSE_JSON + EVAL_IMAGE_PATH                [first train/ trajectory of DATA_BASE]
#     EVAL_POSE_JSON_UNSEEN + EVAL_IMAGE_PATH_UNSEEN  [first test/ trajectory of DATA_BASE]
#     EVAL_PROMPT, EVAL_GPUS [= CUDA_VISIBLE_DEVICES]; set IN_TRAINING_EVAL=0 to disable
#   WANDB_MODE [online]  TRACKER_PROJECT_NAME [dynatoken]
#
# Example:
#   DATA_BASE=outputs/training_data/bird bash dynatokens/train.sh
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${DATA_BASE:?set DATA_BASE to a prepared training-data dir}"
DATA_BASE="$(cd "$DATA_BASE" && pwd)"
DATA_JSON_PATH=${DATA_JSON_PATH:-$DATA_BASE/train_all_textprompt.json}
scene_field() { python -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$DATA_BASE/scene.json" "$1"; }
if [[ -z "${WINDOW_FRAMES:-}" ]]; then
    [[ -f "$DATA_BASE/scene.json" ]] || { echo "ERROR: set WINDOW_FRAMES (no $DATA_BASE/scene.json)" >&2; exit 1; }
    WINDOW_FRAMES=$(scene_field window_frames)
fi

# Default in-training eval trajectories: the first seen (train/) and unseen (test/) pose.
if [[ "${IN_TRAINING_EVAL:-1}" == 1 && -f "$DATA_BASE/scene.json" ]]; then
    first_pose() { ls "$DATA_BASE/$1"/*/pose.json 2>/dev/null | head -1 || true; }
    image="$(scene_field image_path)"
    if [[ -z "${EVAL_POSE_JSON:-}" ]]; then
        EVAL_POSE_JSON="$(first_pose train)"
        if [[ -n "$EVAL_POSE_JSON" ]]; then
            gt="$(scene_field scene_dir)/train/$(basename "$(dirname "$EVAL_POSE_JSON")")/$(scene_field video_name)"
            EVAL_IMAGE_PATH="${EVAL_IMAGE_PATH:-$([[ -f "$gt" ]] && echo "$gt" || echo "$image")}"
        fi
    fi
    if [[ -z "${EVAL_POSE_JSON_UNSEEN:-}" ]]; then
        EVAL_POSE_JSON_UNSEEN="$(first_pose test)"
        EVAL_IMAGE_PATH_UNSEEN="${EVAL_IMAGE_PATH_UNSEEN:-$image}"
    fi
elif [[ "${IN_TRAINING_EVAL:-1}" == 0 ]]; then
    EVAL_POSE_JSON="" EVAL_POSE_JSON_UNSEEN=""
fi
NUM_FRAMES=$(( (WINDOW_FRAMES - 1) * 4 + 1 ))

LEARNING_RATE=${LEARNING_RATE:-3e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
CROSSATTN_TOKEN_DIM=${CROSSATTN_TOKEN_DIM:-64}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-3000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-500}
# The in-training eval reads the checkpoint saved at the same step, so keep these equal.
VALIDATION_STEPS=${VALIDATION_STEPS:-$CHECKPOINTING_STEPS}
NUM_GPUS=${NUM_GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
EVAL_GPUS=${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}

if [[ -z "${RUN_NAME:-}" ]]; then
    RUN_NAME="$(basename "$DATA_BASE")_$(python -c 'import secrets; print(secrets.token_hex(2))')"
fi
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUTS_ROOT/runs/$RUN_NAME}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
export EVAL_START_PORT=${EVAL_START_PORT:-$((MASTER_PORT + 1))}

echo "══════════════════════════════════════════════════════════════════════"
echo "  RUN_NAME:    $RUN_NAME"
echo "  DATA:        $DATA_JSON_PATH"
echo "  WINDOW:      $WINDOW_FRAMES latents ($NUM_FRAMES frames)"
echo "  LR / WD:     $LEARNING_RATE / $WEIGHT_DECAY   token_dim=$CROSSATTN_TOKEN_DIM"
echo "  STEPS:       $MAX_TRAIN_STEPS (ckpt every $CHECKPOINTING_STEPS)"
echo "  OUTPUT_DIR:  $OUTPUT_DIR"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo "══════════════════════════════════════════════════════════════════════"

training_args=(
  --json_path "$DATA_JSON_PATH"
  --neg_prompt_path "$DATA_BASE/shared/hunyuan_neg_prompt.pt"
  --neg_byt5_path "$DATA_BASE/shared/hunyuan_neg_byt5_prompt.pt"
  --causal
  --action
  --i2v_rate 0.2
  --train_time_shift 1.0
  --window_frames "$WINDOW_FRAMES"
  --tracker_project_name "${TRACKER_PROJECT_NAME:-dynatoken}"
  --wandb_run_name "$RUN_NAME"
  --output_dir "$OUTPUT_DIR"
  --max_train_steps "$MAX_TRAIN_STEPS"
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_height 480
  --num_width 832
  --num_frames "$NUM_FRAMES"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --temporal_crossattn_per_block_training true
  --temporal_embed_max_frames "$WINDOW_FRAMES"
  --temporal_crossattn_token_dim "$CROSSATTN_TOKEN_DIM"
  --validation_steps "$VALIDATION_STEPS"
  --eval_action_ckpt "$ACTION_CKPT"
  --eval_gpus "$EVAL_GPUS"
  ${EVAL_POSE_JSON:+--eval_pose_json "$EVAL_POSE_JSON"}
  ${EVAL_IMAGE_PATH:+--eval_image_path "$EVAL_IMAGE_PATH"}
  ${EVAL_POSE_JSON_UNSEEN:+--eval_pose_json_unseen "$EVAL_POSE_JSON_UNSEEN"}
  ${EVAL_IMAGE_PATH_UNSEEN:+--eval_image_path_unseen "$EVAL_IMAGE_PATH_UNSEEN"}
  ${EVAL_PROMPT:+--eval_prompt "$EVAL_PROMPT"}
)

parallel_args=(
  --num_gpus "$NUM_GPUS"
  --sp_size "$NUM_GPUS"
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim "$NUM_GPUS"
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir "$MODEL_PATH/transformer/480p_i2v"
  --ar_action_load_from_dir "$ACTION_CKPT"
  --model_path "$MODEL_PATH"
  --pretrained_model_name_or_path "$MODEL_PATH"
)

optimizer_args=(
  --learning_rate "$LEARNING_RATE"
  --mixed_precision "bf16"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --weight_decay "$WEIGHT_DECAY"
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --dataloader_num_workers 0
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "bf16"
  --enable_gradient_checkpointing_type "full"
  --num_euler_timesteps 50
  --ema_start_step 0
)

cd "$HYWP_ROOT"
torchrun \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes 1 \
    trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${miscellaneous_args[@]}"
