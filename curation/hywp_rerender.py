#!/usr/bin/env python3
"""Step 3: HYWP camera re-rendering — apply camera actions to state frames.

For each (instance, pose, state_i), runs HYWP with the state_i frame as initial
image, a neutral prompt, and the first i camera actions (padded to valid HYWP
length). Extracts the frame at the original camera boundary.

Supports multi-GPU parallelism: each GPU process handles a shard of poses.

Expects PYTHONPATH to include model/hywp/ (set by the shell wrapper).

Usage:
    CUDA_VISIBLE_DEVICES=0 python hywp_rerender.py \
        --instances_json data/instances.json \
        --output_root  data/pipeline_output \
        --model_path   /path/to/HunyuanVideo-1.5 \
        --action_ckpt  /path/to/ar_rl_model/diffusion_pytorch_model.safetensors \
        --gpu_id 0 \
        --num_gpus 4 \
        [--max_instances N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from keyframe_utils import (
    N_STEPS,
    load_poses,
    instance_name,
    instance_output_dir,
    pad_pose,
)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
NEUTRAL_PROMPT = "A static scene."


def build_hywp_args(cli_args, video_length):
    """Build an argparse.Namespace that mimics generate.py's expected args."""
    return argparse.Namespace(
        model_path=cli_args.model_path,
        action_ckpt=cli_args.action_ckpt,
        resolution="480p",
        video_length=video_length,
        seed=cli_args.seed,
        rewrite=False,
        sr=False,
        save_pre_sr_video=False,
        few_step=False,
        model_type="ar",
        width=832,
        height=480,
        aspect_ratio="16:9",
        num_inference_steps=cli_args.num_inference_steps,
        negative_prompt="",
        offloading=True,
        group_offloading=None,
        dtype="bf16",
        enable_torch_compile=False,
        with_ui=False,
        use_sageattn=False,
        sage_blocks_range="0-53",
        use_vae_parallel=False,
        use_fp8_gemm=False,
        quant_type="fp8-per-block",
        include_patterns="double_blocks",
        image_path=None,
        prompt=None,
        pose=None,
        poses_file=None,
        output_path=None,
    )


def extract_frame(video_path, frame_idx, output_path):
    """Extract a single 0-indexed frame from a video."""
    subprocess.run(
        [FFMPEG, "-i", video_path,
         "-vf", f"select=eq(n\\,{frame_idx})",
         "-frames:v", "1",
         output_path,
         "-y", "-loglevel", "error"],
        check=True,
    )


def collect_work(instances, output_root, gpu_id, num_gpus, n_steps, poses_json, max_poses=None):
    """Collect all (instance, pose_index, state_index) work items for this GPU.

    Shards by pose_index across GPUs (round-robin). Groups by video_length
    to minimize HYWP pipeline reloads.

    Returns list of dicts sorted by video_length (ascending), each containing:
        inst, pose_idx, pose_str, pose_name, state_idx,
        padded_pose, video_length, target_frame,
        state_image, output_frame, output_dir
    """
    poses, names = load_poses(poses_json, max_poses, n_steps=n_steps)
    work = []
    for inst in instances:
        inst_dir = instance_output_dir(output_root, inst)
        states_dir = os.path.join(inst_dir, "states")

        # Check that state keyframes exist
        if not os.path.isfile(os.path.join(states_dir, f"state_{n_steps}.png")):
            name = instance_name(inst)
            print(f"  [{name}] States not found, skipping. "
                  f"Run generate_states.py first.")
            continue

        for pose_idx, (pose_str, pose_name) in enumerate(zip(poses, names)):
            pose_dir = os.path.join(inst_dir, pose_name)
            rerender_dir = os.path.join(pose_dir, "hywp_rerender")

            for state_idx in range(1, n_steps + 1):
                output_frame = os.path.join(rerender_dir,
                                            f"state_{state_idx}_view.png")

                # Skip if already done
                if os.path.isfile(output_frame):
                    continue

                padded_pose, video_length, target_frame = pad_pose(
                    pose_str, state_idx
                )
                deblurred = os.path.join(states_dir,
                                         f"state_{state_idx}_deblurred.png")
                state_image = deblurred if os.path.isfile(deblurred) else \
                              os.path.join(states_dir, f"state_{state_idx}.png")

                work.append({
                    "inst": inst,
                    "pose_idx": pose_idx,
                    "pose_str": pose_str,
                    "pose_name": pose_name,
                    "state_idx": state_idx,
                    "padded_pose": padded_pose,
                    "video_length": video_length,
                    "target_frame": target_frame,
                    "state_image": state_image,
                    "output_frame": output_frame,
                    "output_dir": rerender_dir,
                })

    # Deduplicate: group by (state_image, padded_pose) — same input produces
    # identical output, so run HYWP once and copy to duplicate destinations.
    seen = {}  # (state_image, padded_pose) -> primary work item
    deduped = []
    for item in work:
        key = (item["state_image"], item["padded_pose"])
        if key in seen:
            seen[key].setdefault("copies", []).append(item["output_frame"])
        else:
            item["copies"] = []
            seen[key] = item
            deduped.append(item)

    # Sort by video_length to minimize pipeline reloads
    deduped.sort(key=lambda w: w["video_length"])

    # Shard by work item index (after dedup) so GPUs get even load
    # regardless of which poses happen to be missing.
    sharded = [item for i, item in enumerate(deduped) if i % num_gpus == gpu_id]
    return sharded


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances_json", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_path", required=True,
                        help="Path to HunyuanVideo-1.5")
    parser.add_argument("--action_ckpt", required=True,
                        help="Path to action checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="This GPU's index (0-based)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of GPUs (for sharding)")
    parser.add_argument("--poses_json", required=True,
                        help='Camera trajectories: {"train": {name: pose}, "test": {name: pose}}')
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="HYWP sampling steps (default: 30)")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--max_poses", type=int, default=None,
                        help="Limit to first N train poses + all test (for testing)")
    cli_args = parser.parse_args()

    with open(cli_args.instances_json) as f:
        instances = json.load(f)
    if cli_args.max_instances:
        instances = instances[:cli_args.max_instances]

    print(f"GPU {cli_args.gpu_id}/{cli_args.num_gpus}: "
          f"Loaded {len(instances)} instances")

    # Collect work for this GPU
    work = collect_work(instances, cli_args.output_root,
                        cli_args.gpu_id, cli_args.num_gpus, cli_args.n_steps,
                        cli_args.poses_json, cli_args.max_poses)

    if not work:
        print(f"GPU {cli_args.gpu_id}: Nothing to do.")
        return

    print(f"GPU {cli_args.gpu_id}: {len(work)} re-renders to process")

    # Import HYWP (triggers CUDA init)
    from hyvideo.generate import load_pipeline, run_single
    from hyvideo.commons.infer_state import initialize_infer_state

    # Track current pipeline video_length to reload only when needed
    current_video_length = None
    pipe = task = enable_sr = None
    gen_args = None

    for i, item in enumerate(work):
        name = instance_name(item["inst"])
        label = (f"GPU {cli_args.gpu_id} [{i+1}/{len(work)}] "
                 f"{name}/{item['pose_name']}/state_{item['state_idx']}")

        print(f"\n{label}")
        print(f"  Padded pose: {item['padded_pose']} "
              f"(video_length={item['video_length']}, "
              f"extract frame {item['target_frame']})")

        # Reload pipeline if video_length changed
        if item["video_length"] != current_video_length:
            current_video_length = item["video_length"]
            print(f"  Loading HYWP pipeline (video_length={current_video_length}) ...")
            gen_args = build_hywp_args(cli_args, current_video_length)
            # Need a dummy image_path for load_pipeline to detect i2v mode
            gen_args.image_path = item["state_image"]
            initialize_infer_state(gen_args)
            pipe, task, enable_sr = load_pipeline(gen_args)
            print(f"  Pipeline loaded.")

        # Set per-item args
        gen_args.image_path = item["state_image"]
        gen_args.prompt = NEUTRAL_PROMPT

        # Run HYWP
        os.makedirs(item["output_dir"], exist_ok=True)
        gen_video = os.path.join(item["output_dir"],
                                 f"state_{item['state_idx']}_gen.mp4")
        tmp_dir = os.path.join(item["output_dir"],
                               f"_tmp_state_{item['state_idx']}")
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"  Running HYWP ...")
        run_single(pipe, task, enable_sr, gen_args,
                   item["padded_pose"], tmp_dir)

        # Find the generated video
        tmp_gen = os.path.join(tmp_dir, "gen.mp4")
        if not os.path.isfile(tmp_gen):
            print(f"  ERROR: gen.mp4 not found in {tmp_dir}")
            continue

        # Extract target frame
        extract_frame(tmp_gen, item["target_frame"], item["output_frame"])
        print(f"  Extracted frame {item['target_frame']} -> {item['output_frame']}")

        # Copy to duplicate destinations
        for copy_path in item.get("copies", []):
            os.makedirs(os.path.dirname(copy_path), exist_ok=True)
            shutil.copy2(item["output_frame"], copy_path)
            print(f"  Copied (dedup) -> {copy_path}")

        # Save the video and clean up tmp dir
        os.rename(tmp_gen, gen_video)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nGPU {cli_args.gpu_id}: All done. Processed {len(work)} re-renders.")


if __name__ == "__main__":
    main()
