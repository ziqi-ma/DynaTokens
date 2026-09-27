#!/usr/bin/env python3
"""Interpolate between edited keyframes using Kling.

For each consecutive pair of keyframes, generates a Kling video that
smoothly transitions from the start frame to the end frame.

Duration = max(3, min(4, (end_frame - start_frame) // 16)) seconds.

Reads the latest edited image from frames_edited/ for each keyframe,
falling back to original frames if no edited version exists.

Videos are saved to --output-dir (default: videos/ inside --dir) as
interp_{start_frame:04d}_{end_frame:04d}.mp4.

If a video already exists it is skipped. Prompts are cached so re-runs
do not call Gemini again.

Usage:
    python interpolate_keyframes.py \
        --dir data/scene \
        --frames-dir data/scene/frames \
        [--edited-dir data/scene/frames_edited] \
        [--output-dir data/scene/videos] \
        [--description "a ball rolling down a ramp"]

Requires: GOOGLE_API_KEY and FAL_KEY environment variables.
"""

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

FPS = 16
MIN_DURATION = 3
MAX_DURATION = 4
MIN_KLING_DIM = 300
TEXT_MODELS = ["gemini-3.1-pro-preview", "gemini-2.5-pro"]

INTERP_PROMPT_TEMPLATE = """\
{description_block}\
The first image shows a scene at one moment in a video. The second image shows \
the same scene a few seconds later.
{camera_block}\
Write a concise video generation prompt describing the smooth physical transition \
from the first image to the second. Focus on what changes (objects moving, state \
changes, etc.){camera_instruction}. Note that this interpolated clip covers \
only a short segment of the overall process — it must not hallucinate new actions \
or objects that are not visible in the two frames. Be very concise.

Put your prompt on the last line in this format:
Prompt: ...\
"""

CAMERA_ACTION_DESC = {
    "w":     "moves forward (dolly in / zoom in)",
    "s":     "moves backward (dolly out / zoom out)",
    "a":     "strafes left",
    "d":     "strafes right",
    "left":  "rotates left (pan left)",
    "right": "rotates right (pan right)",
    "up":    "tilts upward",
    "down":  "tilts downward",
}

RATE_LIMIT_WAIT = 20


def _make_gemini_client():
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def _generate_prompt(gemini, start_path, end_path, description=None, camera_action="", max_retries=3):
    description_block = f'Overall process: "{description}"\n\n' if description else ""
    cam_desc = CAMERA_ACTION_DESC.get(camera_action, camera_action) if camera_action else ""
    if cam_desc:
        camera_block = f'\nCamera action for this clip: the camera {cam_desc} smoothly throughout.\n\n'
        camera_instruction = f", and explicitly include that the camera {cam_desc} smoothly"
    else:
        camera_block = ""
        camera_instruction = ", and how the camera moves (omit if stationary)"
    template = INTERP_PROMPT_TEMPLATE.format(
        description_block=description_block,
        camera_block=camera_block,
        camera_instruction=camera_instruction,
    )
    pil1 = Image.open(start_path).convert("RGB")
    pil2 = Image.open(end_path).convert("RGB")
    for model in TEXT_MODELS:
        for _ in range(max_retries):
            try:
                resp = gemini.models.generate_content(model=model, contents=[template, pil1, pil2])
                text = (getattr(resp, "text", "") or "").strip()
                parts = re.split(r"\*{0,2}Prompt:\*{0,2}", text)
                prompt = parts[-1].strip() if len(parts) > 1 else text
                if cam_desc and "camera" not in prompt.lower():
                    prompt = prompt.rstrip(".") + f". The camera {cam_desc} smoothly."
                return prompt
            except Exception as e:
                print(e)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Rate limited on {model}, waiting {RATE_LIMIT_WAIT}s ...")
                    time.sleep(RATE_LIMIT_WAIT)
                else:
                    raise
        print(f"  Quota exhausted on {model}, trying next model ...")
    raise RuntimeError("Prompt generation failed: all models quota-exceeded.")


def _resize_to_target(path, target_size, output_dir):
    """Resize image to target_size (w, h) if it differs. Saves to output_dir."""
    img = Image.open(path).convert("RGB")
    if img.size == target_size:
        return path
    resized = img.resize(target_size, Image.LANCZOS)
    out_path = os.path.join(output_dir, os.path.basename(path))
    resized.save(out_path)
    return out_path


def _detect_target_resolution(data_dir, edited_dir):
    """Detect the target resolution for interpolation.

    Checks gen.mp4 first (original rollout), then falls back to the first
    edited frame in frames_edited/.
    """
    # Try gen.mp4
    gen_mp4 = os.path.join(data_dir, "gen.mp4")
    if os.path.isfile(gen_mp4):
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x", gen_mp4
            ], text=True).strip()
            w, h = out.split("x")
            return (int(w), int(h))
        except Exception:
            pass

    # Fall back to first edited frame
    if os.path.isdir(edited_dir):
        for fname in sorted(os.listdir(edited_dir)):
            if fname.startswith("edited_") and fname.endswith(".png"):
                try:
                    img = Image.open(os.path.join(edited_dir, fname))
                    return img.size
                except Exception:
                    pass

    return None


def _ensure_min_size(path):
    img = Image.open(path)
    w, h = img.size
    if w >= MIN_KLING_DIM and h >= MIN_KLING_DIM:
        return path
    scale = max(MIN_KLING_DIM / w, MIN_KLING_DIM / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    resized_path = path.rsplit(".", 1)[0] + "_resized.png"
    img.save(resized_path)
    print(f"    Resized {os.path.basename(path)} from {w}x{h} to {new_w}x{new_h}")
    return resized_path


def _upload_image(path):
    """Upload an image to fal CDN (must be called from main thread)."""
    import fal_client
    path = _ensure_min_size(path)
    return fal_client.upload_file(path)


def _generate_video_kling(prompt, start_url, end_url, duration, save_path):
    import fal_client
    import requests

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"    [kling] {log['message']}")

    result = fal_client.subscribe(
        "fal-ai/kling-video/o3/standard/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": start_url,
            "end_image_url": end_url,
            "duration": str(duration),
            "multi_prompt": None,
            "shot_type": "customize",
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    video_url = result["video"]["url"]
    resp = requests.get(video_url)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)


def _find_frame_image(kf, frames_dir, edited_dir):
    """Return the best available image path for a keyframe."""
    frame_idx = kf["frame"]
    if kf.get("type") == "initial":
        # For initial frame, check edited first (in our pipeline, state_0 is there)
        ep = os.path.join(edited_dir, f"edited_{frame_idx:04d}.png")
        if os.path.isfile(ep):
            return ep
        return os.path.join(frames_dir, f"frame_{frame_idx + 1:04d}.png")
    ep = os.path.join(edited_dir, f"edited_{frame_idx:04d}.png")
    if os.path.isfile(ep):
        return ep
    return os.path.join(frames_dir, f"frame_{frame_idx + 1:04d}.png")


def run(data_dir, frames_dir, edited_dir, output_dir, description=None):
    v0 = os.path.join(data_dir, "merged_keyframes_v0.json")
    initial = os.path.join(data_dir, "merged_keyframes_initial.json")
    input_path = v0 if os.path.isfile(v0) else initial

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No keyframes file found in {data_dir}")

    with open(input_path) as f:
        keyframes = json.load(f)

    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Detect target resolution and resize frames before interpolation
    target_res = _detect_target_resolution(data_dir, edited_dir)
    resized_dir = None
    if target_res:
        resized_dir = os.path.join(output_dir, "_resized_frames")
        os.makedirs(resized_dir, exist_ok=True)
        print(f"Target resolution: {target_res[0]}x{target_res[1]} — will resize frames before interpolation")
    else:
        print("Could not detect target resolution — using frames as-is")

    # Phase 1: collect segments needing work
    segments = []
    for i in range(len(keyframes) - 1):
        kf_start = keyframes[i]
        kf_end = keyframes[i + 1]

        frame_start = kf_start["frame"]
        frame_end = kf_end["frame"]
        ts_start = kf_start["timestamp"]
        ts_end = kf_end["timestamp"]
        duration = min(MAX_DURATION, max(MIN_DURATION, (frame_end - frame_start) // FPS))

        save_path = os.path.join(output_dir, f"interp_{frame_start:04d}_{frame_end:04d}.mp4")
        if os.path.isfile(save_path):
            print(f"  [{ts_start} -> {ts_end}] ({duration}s): already exists, skipping")
            continue

        start_img = _find_frame_image(kf_start, frames_dir, edited_dir)
        end_img = _find_frame_image(kf_end, frames_dir, edited_dir)

        if not os.path.isfile(start_img):
            print(f"  [{ts_start} -> {ts_end}]: missing start image, skipping")
            continue
        if not os.path.isfile(end_img):
            print(f"  [{ts_start} -> {ts_end}]: missing end image, skipping")
            continue

        # Resize to target resolution before interpolation
        if target_res and resized_dir:
            start_img = _resize_to_target(start_img, target_res, resized_dir)
            end_img = _resize_to_target(end_img, target_res, resized_dir)

        camera_action = kf_end.get("action", "") if kf_end.get("type") != "initial" else ""
        stem = f"interp_{frame_start:04d}_{frame_end:04d}"
        prompt_file = os.path.join(output_dir, f"{stem}_prompt.txt")

        cached_prompt = None
        if os.path.isfile(prompt_file):
            cached_prompt = open(prompt_file).read().strip()
            print(f"  [{ts_start} -> {ts_end}] ({duration}s): reusing prompt: {cached_prompt[:80]}...")

        segments.append({
            "start_img": start_img,
            "end_img": end_img,
            "duration": duration,
            "save_path": save_path,
            "label": f"[{ts_start} -> {ts_end}]",
            "camera_action": camera_action,
            "prompt_file": prompt_file,
            "cached_prompt": cached_prompt,
        })

    if not segments:
        print("All segments already exist or were skipped.")
        return

    # Phase 2: generate Gemini prompts in parallel for segments that need them
    needs_prompt = [s for s in segments if s["cached_prompt"] is None]
    if needs_prompt:
        gemini = _make_gemini_client()
        print(f"\nGenerating {len(needs_prompt)} prompt(s) in parallel ...")

        def _gen_prompt(seg):
            cam_desc = CAMERA_ACTION_DESC.get(seg["camera_action"], seg["camera_action"]) if seg["camera_action"] else ""
            print(f"  {seg['label']}: generating prompt (camera: {cam_desc or 'unknown'}) ...")
            prompt = _generate_prompt(gemini, seg["start_img"], seg["end_img"],
                                      description, camera_action=seg["camera_action"])
            print(f"  {seg['label']}: -> {prompt[:80]}...")
            with open(seg["prompt_file"], "w") as f:
                f.write(prompt)
            seg["cached_prompt"] = prompt

        with ThreadPoolExecutor(max_workers=len(needs_prompt)) as pool:
            futures = {pool.submit(_gen_prompt, seg): seg for seg in needs_prompt}
            for future in as_completed(futures):
                seg = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  {seg['label']}: prompt generation FAILED — {e}")

    # Remove segments that failed prompt generation
    segments = [s for s in segments if s["cached_prompt"] is not None]
    if not segments:
        print("No segments with valid prompts.")
        return

    # Phase 3: upload images sequentially (fal_client auth isn't thread-safe)
    print(f"\nUploading {len(segments)} image pairs ...")
    upload_cache = {}
    for seg in segments:
        for key in ("start_img", "end_img"):
            path = seg[key]
            if path not in upload_cache:
                upload_cache[path] = _upload_image(path)
        seg["start_url"] = upload_cache[seg["start_img"]]
        seg["end_url"] = upload_cache[seg["end_img"]]

    # Phase 4: generate videos in parallel (Kling API is IO-bound)
    print(f"Generating {len(segments)} video(s) in parallel ...")

    def _gen_one(seg):
        _generate_video_kling(seg["cached_prompt"], seg["start_url"], seg["end_url"],
                              seg["duration"], seg["save_path"])
        return seg

    with ThreadPoolExecutor(max_workers=len(segments)) as pool:
        futures = {pool.submit(_gen_one, seg): seg for seg in segments}
        for future in as_completed(futures):
            seg = futures[future]
            try:
                future.result()
                print(f"  {seg['label']} ({seg['duration']}s): saved {seg['save_path']}")
            except Exception as e:
                print(f"  {seg['label']}: FAILED — {e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, help="Data directory")
    parser.add_argument("--frames-dir", required=True, help="Original frames directory")
    parser.add_argument("--edited-dir", default=None, help="Edited frames directory (default: frames_edited/ inside --dir)")
    parser.add_argument("--output-dir", default=None, help="Output directory for videos (default: videos/ inside --dir)")
    parser.add_argument("--description", default=None, help="Text description of the physical process")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.dir)
    edited_dir = args.edited_dir or os.path.join(data_dir, "frames_edited")
    output_dir = args.output_dir or os.path.join(data_dir, "videos")
    run(data_dir, args.frames_dir, edited_dir, output_dir, args.description)


if __name__ == "__main__":
    main()
