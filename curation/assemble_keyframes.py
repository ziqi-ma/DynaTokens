#!/usr/bin/env python3
"""Assemble keyframes for interpolation and stitching.

After state generation (Step 1) and HYWP re-rendering (Step 3), this script
assembles the keyframe images into the format expected by
interpolate_keyframes.py and stitch_keyframes.py:

  - Builds merged_keyframes_initial.json from the pose action string
  - Creates frames_edited/ with the assembled keyframes:
      edited_0000.png = state_0 (original image, no camera change)
      edited_0024.png = state_1 re-rendered at view_1
      edited_0052.png = state_2 re-rendered at view_2
      edited_0076.png = state_3 re-rendered at view_3
  - Creates frames/ with fallback frames (copies of nearest keyframe)

Then runs interpolation (Kling) and stitching for each pose.

Usage:
    python assemble_keyframes.py \
        --instances_json scenes.json \
        --poses_json     poses.json \
        --output_root    outputs/curation/my_scenes \
        [--workers 8] \
        [--skip_interpolation] \
        [--skip_stitch] \
        [--max_instances N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from keyframe_utils import (
    N_STEPS,
    action_boundaries,
    build_keyframes_json,
    construct_prompt,
    load_poses,
    instance_name,
    instance_output_dir,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERPOLATE = os.path.join(SCRIPT_DIR, "interpolate_keyframes.py")
STITCH = os.path.join(SCRIPT_DIR, "stitch_keyframes.py")

REVIEW_PROMPT = """\
You are reviewing a generated video. Your default answer is PASS — only override to FAIL \
if there is a clear, undeniable problem that would make the video completely unusable.

The video was supposed to show:
"{prompt}"

Evaluate two things:

1. **Text alignment**: PASS unless (a) the main subject is entirely the wrong thing, \
(b) the core action is completely absent, or (c) the subject moves in a clearly wrong \
way. Ignore initial state, timing, partial completion, camera behavior, \
or minor details.

2. **Main object artifacts**: PASS unless the main subject physically splits into multiple \
copies or completely vanishes mid-video. Ignore color shifts, blurriness, shape changes, \
pose imperfections, or any transient glitch.

**Note:** This video intentionally includes camera movement (panning, tilting, dollying). \
Camera motion causing the subject to shift position, exit frame partially, or change \
perspective is expected and must never be flagged.

3. **Physics plausibility**: PASS unless the main process shows clear physics violations.

**Note:** This video intentionally includes camera movement (panning, tilting, dollying). \
Camera motion causing the subject to shift position, exit frame partially, or change \
perspective is expected and must never be flagged.

When in doubt, always PASS.

Respond in this exact format:
Alignment: PASS or FAIL
Artifacts: PASS or FAIL
Feedback: <one sentence, or "None" if both pass>
"""

PY = sys.executable
_print_lock = threading.Lock()


def _review_stitched(mp4_path, prompt, label="", physics=False):
    """Upload stitched.mp4 to Gemini and review against motion_prompt.

    Saves result to stitched_review.json next to the mp4. Skips if already done.
    Returns (passed, feedback).
    """
    import re as _re, time as _time
    review_path = mp4_path.replace(".mp4", "_review.json")
    if os.path.isfile(review_path):
        with open(review_path) as f:
            cached = json.load(f)
        log(label, f"review cached: {'PASS' if cached['passed'] else 'FAIL'} — {cached['feedback'][:80]}")
        return cached["passed"], cached["feedback"]

    from google import genai
    from google.genai import types
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        log(label, "GOOGLE_API_KEY not set, skipping review.")
        return None, None
    client = genai.Client(api_key=api_key)

    prompt_text = REVIEW_PROMPT.format(prompt=prompt)
    log(label, "uploading for review ...")
    upload_path = mp4_path
    if physics:
        # Re-encode at 8fps so Gemini interprets frame timing correctly
        slow_path = mp4_path.replace(".mp4", "_8fps.mp4")
        if not os.path.isfile(slow_path):
            subprocess.run(
                ["ffmpeg", "-i", mp4_path, "-r", "8", "-y",
                 "-loglevel", "error", slow_path],
                check=True,
            )
        upload_path = slow_path
    video_file = client.files.upload(file=upload_path)
    while video_file.state.name == "PROCESSING":
        _time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        client.files.delete(name=video_file.name)
        log(label, "Gemini file processing failed, skipping review.")
        return None, None

    raw = ""
    MODELS = ["gemini-3.1-pro-preview", "gemini-2.5-pro"]
    try:
        for model in MODELS:
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=[
                            prompt_text,
                            types.Part(file_data=types.FileData(
                                file_uri=video_file.uri, mime_type="video/mp4"
                            )),
                        ],
                    )
                    raw = (getattr(resp, "text", "") or "").strip()
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        _time.sleep(20 * (2 ** attempt))
                    else:
                        raise
            if raw:
                break
    finally:
        client.files.delete(name=video_file.name)

    align_pass = bool(_re.search(r"Alignment:\s*PASS", raw, _re.IGNORECASE))
    artifact_pass = bool(_re.search(r"Artifacts:\s*PASS", raw, _re.IGNORECASE))
    passed = align_pass and artifact_pass
    feedback_m = _re.search(r"Feedback:\s*(.+)", raw, _re.DOTALL)
    feedback = feedback_m.group(1).strip() if feedback_m else raw

    result = {"passed": passed, "alignment": align_pass, "artifacts": artifact_pass,
              "feedback": feedback, "raw": raw}
    with open(review_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "PASS" if passed else "FAIL"
    log(label, f"review: {status} — {feedback[:100]}")
    return passed, feedback


def log(prefix, msg):
    with _print_lock:
        print(f"[{prefix}] {msg}", flush=True)


def _run(cmd, check=True):
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def assemble_pose(inst_dir, pose_str, pose_name, states_dir, n_steps):
    """Assemble keyframes for one pose into frames_edited/ and merged_keyframes_initial.json."""
    pose_dir = os.path.join(inst_dir, pose_name)
    edited_dir = os.path.join(pose_dir, "frames_edited")
    frames_dir = os.path.join(pose_dir, "frames")
    rerender_dir = os.path.join(pose_dir, "hywp_rerender")
    keyframes_file = os.path.join(pose_dir, "merged_keyframes_initial.json")

    os.makedirs(edited_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Build keyframes JSON
    keyframes = build_keyframes_json(pose_str.replace(" ", ""))
    with open(keyframes_file, "w") as f:
        json.dump(keyframes, f, indent=2)

    # Assemble edited keyframes
    boundaries = action_boundaries(pose_str.replace(" ", ""))

    # state_0 = original image at frame 0 (no camera change needed)
    state_0_src = os.path.join(states_dir, "state_0.png")
    state_0_dst = os.path.join(edited_dir, "edited_0000.png")
    if os.path.isfile(state_0_src) and not os.path.isfile(state_0_dst):
        shutil.copy2(state_0_src, state_0_dst)

    # state_i at view_i for i=1..N
    for i in range(1, n_steps + 1):
        if i > len(boundaries):
            break
        _, frame = boundaries[i - 1]
        src = os.path.join(rerender_dir, f"state_{i}_view.png")
        dst = os.path.join(edited_dir, f"edited_{frame:04d}.png")
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)

    # Create minimal frames/ directory for stitch fallback
    # Copy state_0 as frame_0001.png (used for resolution detection)
    frame_1 = os.path.join(frames_dir, "frame_0001.png")
    if not os.path.isfile(frame_1) and os.path.isfile(state_0_src):
        shutil.copy2(state_0_src, frame_1)

    return pose_dir


def process_pose(pose_str, pose_name, inst_dir, states_dir, prompt,
                 n_steps, skip_interpolation, skip_stitch, label=""):
    """Assemble, interpolate, and stitch for one pose."""
    pose_dir = os.path.join(inst_dir, pose_name)
    stitched = os.path.join(pose_dir, "stitched.mp4")

    stitched_trim = os.path.join(pose_dir, "stitched_trim8.mp4")
    if os.path.isfile(stitched) and os.path.isfile(stitched_trim):
        log(label, "stitched.mp4 and stitched_trim8.mp4 exist, skipping.")
        return True

    # Check that HYWP re-renders exist
    rerender_dir = os.path.join(pose_dir, "hywp_rerender")
    missing = []
    for i in range(1, n_steps + 1):
        view_path = os.path.join(rerender_dir, f"state_{i}_view.png")
        if not os.path.isfile(view_path):
            missing.append(f"state_{i}_view.png")
    if missing:
        log(label, f"Missing re-renders: {', '.join(missing)}. "
            f"Run hywp_rerender.py first.")
        return False

    # Assemble
    log(label, "[1/3] Assembling keyframes ...")
    assemble_pose(inst_dir, pose_str, pose_name, states_dir, n_steps)

    frames_dir = os.path.join(pose_dir, "frames")
    edited_dir = os.path.join(pose_dir, "frames_edited")

    if skip_interpolation:
        log(label, "Skipping interpolation (--skip_interpolation)")
        return True

    # Interpolate (skip if videos already exist)
    import glob as _glob
    videos_dir = os.path.join(pose_dir, "videos")
    if _glob.glob(os.path.join(videos_dir, "interp_*.mp4")):
        log(label, f"Interpolation videos exist, skipping Kling.")
    else:
        log(label, "[2/3] Interpolating ...")
        ok = _run([PY, INTERPOLATE,
                   "--dir", pose_dir,
                   "--frames-dir", frames_dir,
                   "--edited-dir", edited_dir,
                   "--description", prompt], check=False)
        if not ok:
            log(label, "Interpolation failed.")
            return False
        if not _glob.glob(os.path.join(videos_dir, "interp_*.mp4")):
            log(label, "No interpolation videos produced, skipping stitch.")
            return False

    if skip_stitch:
        log(label, "Skipping stitch (--skip_stitch)")
        return True

    # Stitch — original (no trim)
    log(label, "[3/3] Stitching (original) ...")
    _run([PY, STITCH,
          "--dir", pose_dir,
          "--frames-dir", frames_dir,
          "--output", stitched])

    # Stitch — trimmed version
    stitched_trim = os.path.join(pose_dir, "stitched_trim8.mp4")
    log(label, "[3/3] Stitching (trim=8) ...")
    _run([PY, STITCH,
          "--dir", pose_dir,
          "--frames-dir", frames_dir,
          "--output", stitched_trim,
          "--trim", "8"])

    log(label, f"Done: {stitched}, {stitched_trim}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances_json", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--poses_json", required=True,
                        help='Camera trajectories: {"train": {name: pose}, "test": {name: pose}}')
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--workers", type=int, default=4,
                        help="Poses to process in parallel (default: 4)")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--max_poses", type=int, default=None,
                        help="Limit to first N train poses + all test (for testing)")
    parser.add_argument("--skip_interpolation", action="store_true")
    parser.add_argument("--skip_stitch", action="store_true")
    parser.add_argument("--physics", action="store_true", default=True,
                        help="Pass fps=8 to Gemini when reviewing (default: True)")
    args = parser.parse_args()

    with open(args.instances_json) as f:
        instances = json.load(f)
    if args.max_instances:
        instances = instances[:args.max_instances]

    print(f"Loaded {len(instances)} instances from {args.instances_json}")

    for inst_idx, inst in enumerate(instances):
        name = instance_name(inst)
        prompt = construct_prompt(inst)

        print(f"\n{'='*60}")
        print(f"[{inst_idx+1}/{len(instances)}] {name}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"{'='*60}")

        inst_dir = instance_output_dir(args.output_root, inst)
        states_dir = os.path.join(inst_dir, "states")

        if not os.path.isdir(states_dir):
            print(f"  States not found at {states_dir}. Run generate_states.py first.")
            continue

        def _pose_task(pose_str, name_path):
            pose_basename = os.path.basename(name_path)
            split = os.path.dirname(name_path)
            label = f"{name}/{split}/{pose_basename}"
            return process_pose(
                pose_str, name_path, inst_dir, states_dir, prompt,
                args.n_steps, args.skip_interpolation, args.skip_stitch,
                label=label,
            )

        poses, names = load_poses(args.poses_json, args.max_poses, n_steps=args.n_steps)
        print(f"\nProcessing {len(poses)} poses with {args.workers} workers ...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_pose_task, ps, np): np
                       for ps, np in zip(poses, names)}
            for fut in as_completed(futures):
                name_path = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    with _print_lock:
                        print(f"  ERROR [{name_path}]: {e}", flush=True)

    print("\nAll instances complete.")

    # Review phase: review stitched.mp4 for all poses, 20 at a time
    review_tasks = []
    for inst in instances:
        inst_dir = instance_output_dir(args.output_root, inst)
        prompt = construct_prompt(inst)
        poses, names = load_poses(args.poses_json, args.max_poses, n_steps=args.n_steps)
        for name_path in names:
            pose_dir = os.path.join(inst_dir, name_path)
            trim_mp4 = os.path.join(pose_dir, "stitched.mp4")
            review_json = os.path.join(pose_dir, "stitched_review.json")
            if os.path.isfile(trim_mp4) and not os.path.isfile(review_json):
                label = f"{instance_name(inst)}/{name_path}"
                review_tasks.append((trim_mp4, prompt, label))

    if review_tasks:
        print(f"\nReview phase: {len(review_tasks)} video(s) to review (20 workers) ...")
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_review_stitched, mp4, p, lbl, args.physics): lbl
                       for mp4, p, lbl in review_tasks}
            for fut in as_completed(futures):
                lbl = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    with _print_lock:
                        print(f"  ERROR [{lbl}]: {e}", flush=True)
    else:
        print("\nReview phase: all videos already reviewed.")


if __name__ == "__main__":
    main()
