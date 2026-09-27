# hywp — vendored HY-WorldPlay

This directory contains the subset of
[Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
(as of March 2026) that DynaTokens needs for training, inference and curation
(`hyvideo/` inference code, `trainer/` training code). It is distributed under
the Tencent HY-WorldPlay Community License, see [`License.txt`](License.txt);
note its territory restrictions.

## Modifications

DynaTokens-specific:
- `TemporalCrossAttnBlock` and `add_temporal_crossattn_per_block_parameters()` in
  `hyvideo/models/transformers/worldplay_1_5_transformer.py` (inference) and
  `trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py`
  (training): per double-stream block, image queries cross-attend to learned
  temporal tokens; the result is added after the attention residual.
- `trainer/trainer_args.py`: `--temporal_crossattn_per_block_training`,
  `--temporal_embed_max_frames`, `--temporal_crossattn_token_dim`,
  `--neg_prompt_path/--neg_byt5_path`, in-training eval options (`--eval_*`),
  `--optimizer_type`.
- `trainer/pipelines/lora_pipeline.py`: freeze the base model, train only the
  temporal cross-attention branch.
- `trainer/models/loader/component_loader.py`: attach the branch after loading.
- `trainer/training/ar_hunyuan_mem_training_pipeline.py`: SP gradient sync for the
  branch, logging, optional AdamW, `_maybe_eval` hook.
- `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`: in-training video
  eval in a background subprocess (`scripts/inference.py`).
- `trainer/training/training_utils.py`: checkpointing when no parameter is
  FSDP-sharded (trainable weights in safetensors + `optimizer.pt`).
- `scripts/inference.py`: inference with a DynaTokens checkpoint.

General fixes:
- `trainer/.../modules/attention.py`: chunked causal attention without
  materialising the dense O(S²) mask.
- `trainer/.../modules/modulate_layers.py`, `token_refiner.py`, the transformer
  `forward`: batch size > 1.
- `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`: configurable negative
  prompt paths, per-latent-frame pose files, short videos.
- `hyvideo/pipelines/worldplay_video_pipeline.py`, `hyvideo/generate.py`:
  super-resolution offloading fix, `--poses_file` batch mode.
- Minor compatibility fixes (flash-attn `unpad_input`, torch.dynamo config,
  process-group timeout, muon optimizer device placement).

Files of the original repository not used by DynaTokens were removed.
