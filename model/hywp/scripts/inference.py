"""
DynaToken inference: load a trained temporal cross-attention checkpoint on top of
HY-WorldPlay (AR, 480p i2v) and render one or more camera trajectories.

Single job:
    torchrun --nproc_per_node=<N> scripts/inference.py \
        --model_path <HunyuanVideo-1.5 dir> \
        --action_ckpt <HY-WorldPlay ar_rl_model/diffusion_pytorch_model.safetensors> \
        --temporal_embed_ckpt <run>/checkpoint-<step>/transformer/diffusion_pytorch_model.safetensors \
        --pose_json <pose.json> \
        --image_path <first frame .png, or GT .mp4 whose first frame is used> \
        --output_dir <out>

Multiple jobs (model loaded once):
    ... --jobs_json jobs.json
    jobs.json: [{"pose_json": "...", "image_path": "...", "output_dir": "...", "prompt": ""}, ...]

Omit --temporal_embed_ckpt to run the base HY-WorldPlay model.
Writes <output_dir>/gen.mp4 (and metrics.json with frame MSE when --image_path is a GT video).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import numpy as np
import torch
import cv2
import imageio
import einops
from PIL import Image
from safetensors.torch import load_file

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from scipy.spatial.transform import Rotation as R


# ── helpers ──────────────────────────────────────────────────────────────────

mapping = {
    (0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3, (0, 0, 0, 1): 4, (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6, (0, 1, 1, 0): 7, (0, 1, 0, 1): 8,
}

def one_hot_to_label(one_hot):
    return torch.tensor([mapping[tuple(r.tolist())] for r in one_hot])

def pose_json_to_inputs(pose_json_path):
    """
    Convert training-format pose.json (keys: 'w2c', 'intrinsic') to
    (w2c_tensor, intrinsic_tensor, action_tensor) suitable for the pipeline.
    """
    pose_json = json.load(open(pose_json_path))
    keys = list(pose_json.keys())
    n = len(keys)

    w2c_list, intrinsic_list = [], []
    for k in keys:
        w2c = np.array(pose_json[k]['w2c'])
        intrinsic = np.array(pose_json[k]['intrinsic'])
        # normalize intrinsic (same as dataset code)
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        w2c_list.append(w2c)
        intrinsic_list.append(intrinsic)

    # camera-center normalization (align first camera to origin)
    w2c_arr = np.array(w2c_list)
    c2w_arr = np.linalg.inv(w2c_arr)
    C0_inv = np.linalg.inv(c2w_arr[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w_arr])
    w2c_arr = np.linalg.inv(c2w_aligned)

    # compute action labels from relative c2w
    c2w_arr = np.linalg.inv(w2c_arr)
    C_inv = np.linalg.inv(c2w_arr[:-1])
    rel_c2w = np.zeros_like(c2w_arr)
    rel_c2w[0] = c2w_arr[0]
    rel_c2w[1:] = C_inv @ c2w_arr[1:]

    trans_one_hot = np.zeros((n, 4), dtype=np.int32)
    rot_one_hot   = np.zeros((n, 4), dtype=np.int32)
    move_thresh = 0.0001
    for i in range(1, n):
        move_dirs  = rel_c2w[i, :3, 3]
        move_norm  = np.linalg.norm(move_dirs)
        rot_angles = R.from_matrix(rel_c2w[i, :3, :3]).as_euler('xyz', degrees=True)
        if move_norm > move_thresh:
            nd = move_dirs / move_norm
            ang = np.arccos(nd.clip(-1, 1)) * 180 / np.pi
            if ang[2] <  60: trans_one_hot[i, 0] = 1
            if ang[2] > 120: trans_one_hot[i, 1] = 1
            if ang[0] <  60: trans_one_hot[i, 2] = 1
            if ang[0] > 120: trans_one_hot[i, 3] = 1
        if rot_angles[1] >  5e-2: rot_one_hot[i, 0] = 1
        if rot_angles[1] < -5e-2: rot_one_hot[i, 1] = 1
        if rot_angles[0] >  5e-2: rot_one_hot[i, 2] = 1
        if rot_angles[0] < -5e-2: rot_one_hot[i, 3] = 1

    trans_label = one_hot_to_label(torch.tensor(trans_one_hot))
    rot_label   = one_hot_to_label(torch.tensor(rot_one_hot))
    action      = trans_label * 9 + rot_label

    return (
        torch.as_tensor(w2c_arr, dtype=torch.float32),
        torch.as_tensor(np.array(intrinsic_list), dtype=torch.float32),
        action,
    )


def extract_first_frame(video_path, out_path):
    """Extract first frame of a video and save as PNG."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read {video_path}"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame_rgb).save(out_path)
    return out_path


def save_video(video, path, fps=16):
    if video.ndim == 5:
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c").cpu().numpy()
    imageio.mimwrite(path, vid, fps=fps)


def run_one_job(pipe, job, local_rank, num_inference_steps, seed):
    pose_json_path = job["pose_json"]
    image_path     = job["image_path"]
    output_dir     = job["output_dir"]
    prompt         = job.get("prompt", "")

    os.makedirs(output_dir, exist_ok=True)

    # extract first frame (rank 0 only, then sync)
    first_frame_path = os.path.join(output_dir, "first_frame.png")
    if image_path.endswith(".mp4") and os.path.exists(image_path):
        if local_rank == 0:
            extract_first_frame(image_path, first_frame_path)
        torch.distributed.barrier()
    else:
        first_frame_path = image_path

    # load pose + actions
    w2c, intrinsics, action = pose_json_to_inputs(pose_json_path)
    n_latents = w2c.shape[0]
    video_length = (n_latents - 1) * 4 + 1
    print(f"Pose: {n_latents} latent frames → {video_length} video frames")

    if local_rank == 0:
        free, total = torch.cuda.mem_get_info()
        print(f"[eval] GPU memory before inference: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    out = pipe(
        enable_sr=False,
        prompt=prompt,
        aspect_ratio="16:9",
        num_inference_steps=num_inference_steps,
        video_length=video_length,
        negative_prompt="",
        seed=seed,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        reference_image=first_frame_path,
        viewmats=w2c.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=False,
        chunk_latent_frames=4,
        model_type="ar",
    )

    if local_rank == 0:
        out_path = os.path.join(output_dir, "gen.mp4")
        save_video(out.videos, out_path)
        print(f"Saved to {out_path}")

        gen = out.videos
        if gen.ndim == 5:
            gen = gen[0]

        gt_frames = []
        if image_path.endswith(".mp4"):
            cap = cv2.VideoCapture(image_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gt_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if gt_frames:
            gen_for_mse = gen[:, 1:]
            n_compare = min(gen_for_mse.shape[1], len(gt_frames))
            gen_for_mse = gen_for_mse[:, :n_compare]
            gt = torch.from_numpy(np.stack(gt_frames[:n_compare], axis=0)).float() / 255.0
            gt = gt.permute(3, 0, 1, 2)

            if gt.shape[-2:] != gen_for_mse.shape[-2:]:
                gt = torch.nn.functional.interpolate(
                    gt.permute(1, 0, 2, 3),
                    size=gen_for_mse.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).permute(1, 0, 2, 3)

            mse = ((gen_for_mse.cpu().float() - gt) ** 2).mean().item()
            print(f"Frame MSE: {mse:.6f}")

            metrics_path = os.path.join(output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"frame_mse": mse}, f)
        else:
            print("No GT video available — skipping MSE computation.")


def load_temporal_crossattn(transformer, ckpt_path):
    """Attach TemporalCrossAttnBlocks to the transformer and load their weights.

    The checkpoint written by training contains only the trainable
    ``temporal_crossattn_blocks.*`` tensors; max_frames and token_dim are read
    from the token table shape.
    """
    sd = {k: v for k, v in load_file(ckpt_path).items() if k.startswith("temporal_crossattn_blocks.")}
    assert sd, f"No temporal_crossattn_blocks.* weights found in {ckpt_path}"
    w0 = sd["temporal_crossattn_blocks.0.temporal_tokens.weight"]
    max_frames, token_dim = w0.shape
    transformer.add_temporal_crossattn_per_block_parameters(max_frames=max_frames, token_dim=token_dim)
    device = next(transformer.parameters()).device
    blocks = transformer.temporal_crossattn_blocks.to(device=device, dtype=w0.dtype)
    blocks.load_state_dict({k[len("temporal_crossattn_blocks."):]: v for k, v in sd.items()}, strict=True)
    print(f"Loaded temporal_crossattn_blocks: {len(blocks)} blocks, "
          f"max_frames={max_frames}, token_dim={token_dim}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",          type=str, required=True)
    parser.add_argument("--action_ckpt",         type=str, required=True)
    parser.add_argument("--temporal_embed_ckpt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed",                type=int, default=42)
    # multi-job mode
    parser.add_argument("--jobs_json",  type=str, default=None,
                        help="Path to JSON file with list of {pose_json, image_path, output_dir, prompt} dicts")
    # single-job mode
    parser.add_argument("--pose_json",  type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prompt",     type=str, default="")
    parser.add_argument("--enable_group_offloading", action="store_true", default=False)
    args = parser.parse_args()

    # build job list
    if args.jobs_json is not None:
        jobs = json.load(open(args.jobs_json))
    else:
        assert args.pose_json and args.image_path and args.output_dir, \
            "Provide either --jobs_json or --pose_json/--image_path/--output_dir"
        jobs = [{"pose_json": args.pose_json, "image_path": args.image_path,
                 "output_dir": args.output_dir, "prompt": args.prompt}]

    # ── init distributed ──
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    parallel_dims = initialize_parallel_state(sp=world_size)
    torch.cuda.set_device(local_rank)

    infer_args = argparse.Namespace(
        chunk_latent_frames=4,
        model_type="ar",
        attn_type="torch_causal",
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        use_fp8_gemm=False,
        quant_type="fp8-per-block",
        use_vae_parallel=False,
    )
    initialize_infer_state(infer_args)

    # ── build pipeline (once) ──
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=args.enable_group_offloading,
        enable_group_offloading=args.enable_group_offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=args.action_ckpt,
    )

    # ── load DynaToken temporal cross-attention weights (once) ──
    if args.temporal_embed_ckpt is not None:
        load_temporal_crossattn(pipe.transformer, args.temporal_embed_ckpt)

    # ── run all jobs ──
    for i, job in enumerate(jobs):
        print(f"\n{'='*60}")
        print(f"Job {i+1}/{len(jobs)}: {job['output_dir']}")
        print(f"{'='*60}")
        run_one_job(pipe, job, local_rank, args.num_inference_steps, args.seed)

    if local_rank == 0:
        print(f"\nAll {len(jobs)} jobs complete.")


if __name__ == "__main__":
    main()
