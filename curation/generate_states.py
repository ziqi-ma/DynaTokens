#!/usr/bin/env python3
"""Step 1: Generate static-camera dynamics video and extract state keyframes.

For each instance, generates a video showing the object dynamics with a fixed
camera, then extracts N+1 evenly-spaced state keyframes (state_0 = original
image, state_1..N from the generated video).

Includes a reflection loop: after keyframe extraction, Gemini checks for
coherent motion and static background. If either fails, Gemini refines the
prompt and the video is regenerated (up to --max_retries attempts).

Supports multiple video generation backends:
  - veo:  Google Veo 3.1 (requires GOOGLE_API_KEY)
  - kling: Kling 3.0 via fal.ai (requires FAL_KEY)

Usage:
    python generate_states.py \
        --instances_json data/instances.json \
        --data_root  data/ \
        --output_root  data/pipeline_output \
        --video_model  kling \
        [--duration 5] \
        [--n_steps 3] \
        [--max_instances N] \
        [--skip_check]
"""

import argparse
import json
import mimetypes
import os
import re
import subprocess
import time

from keyframe_utils import (
    N_STEPS,
    construct_prompt,
    instance_image_path,
    instance_name,
    instance_output_dir,
)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

VIDEO_PROMPT_SUFFIX = "Beside the above stated, all other elements should remain static. The background and lighting must be preserved throughout the video and must not change. The camera angle must remain completely static and must not move at all. Do not zoom in, zoom out, or rotate the camera."
TARGET_ASPECT = 16 / 9  # HYWP native aspect ratio

HYWP_RESOLUTION = (832, 480)  # HYWP native resolution


def _center_crop_and_resize(image_path, save_path):
    """Center-crop to 16:9 and resize to HYWP native resolution (832x480)."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    current_aspect = w / h

    if current_aspect > TARGET_ASPECT:
        # Too wide — crop width
        new_w = round(h * TARGET_ASPECT)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_aspect < TARGET_ASPECT - 0.01:
        # Too tall — crop height
        new_h = round(w / TARGET_ASPECT)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize(HYWP_RESOLUTION, Image.LANCZOS)
    img.save(save_path)
    print(f"    Cropped+resized {w}x{h} -> {HYWP_RESOLUTION[0]}x{HYWP_RESOLUTION[1]}")
    return save_path


GEMINI_TEXT_MODELS = ["gemini-3.1-pro-preview", "gemini-2.5-pro"]
RATE_LIMIT_WAIT = 20
MAX_REFLECTION_ATTEMPTS = 3

PLAN_WITH_PROMPT_TEMPLATE = """\
You are a director planning keyframes for a video generation pipeline. You \
are given an initial frame and an action description. Your job is to plan \
the intermediate keyframes and write a video generation prompt.

Action description: "{action_prompt}"

Look at the image carefully. Then:

1. **Identify** the main subject and describe what you see in the initial frame.
2. **Plan keyframes** — describe the scene at the start, midpoint, and end \
of the action. Focus on the subject's position, pose, and state at each moment.
3. **Write** a video generation prompt that explicitly describes the \
progression through all 3 keyframes as a continuous motion. The prompt must \
mention the starting position/state, the intermediate position/state, and \
the final position/state. Ignore the background — focus only on the subject's \
motion or physical change.

Respond in this exact format:

Subject: <the main subject and what you see in the initial frame>
Keyframe 0: <the initial state — what you see in the image>
Keyframe 1: <the midpoint of the action>
Keyframe 2: <the final state after the action completes>
Prompt: <a video generation prompt that describes the subject moving/changing through all 3 keyframes>\
"""

PLAN_NO_PROMPT_TEMPLATE = """\
You are a director planning keyframes for a video generation pipeline. You \
are given an initial frame with NO action description. Your job is to predict \
what would physically happen next and plan the keyframes.

Look at the image carefully. Then:

1. **Analyze** the scene — identify subjects, physical setup, unstable objects, \
forces at play, ongoing motion, etc.
2. **Predict** what would physically happen next based on gravity, momentum, \
collisions, fluid dynamics, or other natural forces.
3. **Plan keyframes** — describe the scene at the start, midpoint, and end \
of the predicted process. Focus on the subject's position and state at each moment.
4. **Write** a video generation prompt that describes the \
progression through all 3 keyframes as a continuous motion (without explicitly stating Keyframe 1:, etc). The prompt must \
mention the starting position/state, the intermediate position/state, and \
the final position/state. Ignore the background — focus only on the subject's \
motion or physical change.

Respond in this exact format:

Subject: <the main subject(s) and the physical setup you see>
Keyframe 0: <the initial state — what you see in the image>
Keyframe 1: <the midpoint of the process>
Keyframe 2: <the final state>
Prompt: <a video generation prompt that describes the subject moving/changing through all 3 keyframes>\
"""

REFINE_PROMPT_TEMPLATE = """\
You are refining a video generation prompt. The previous prompt produced a \
video that failed quality checks.

{action_block}\
Previous plan and prompt: "{previous_prompt}"

Feedback from the quality checker:
{feedback}

Look at the initial frame. Revisit the keyframe planning and rewrite the \
prompt to address the feedback. Be more explicit about what should and should \
not change, but be careful to not over correct. Focus only on the key dynamic/physics — ignore the background. \
The prompt must describe the \
progression through all 3 keyframes as a continuous motion (without explicitly stating Keyframe 1:, etc). The prompt must \
mention the starting position/state, the intermediate position/state, and \
the final position/state. Ignore the background — focus only on the subject's \
motion or physical change.

Respond in this exact format:

Subject: <the main subject and what you see in the initial frame>
Keyframe 0: <the initial state — what you see in the image>
Keyframe 1: <the midpoint of the action>
Keyframe 2: <the final state after the action completes>
Prompt: <a video generation prompt that describes the subject moving/changing through all 3 keyframes>\
"""

CHECK_KEYFRAMES_TEMPLATE = """\
You are a quality checker for a video generation pipeline. You are given a \
sequence of keyframes extracted from a generated video. The video was supposed \
to show the following action with a completely static camera:

{action_block}\

Evaluate these keyframes (shown in temporal order) on three criteria:

1. **Coherent motion**: Does the subject actually move/change as described? Because you are only given very few keyframes, there might be jumps or missing intermediates, and that's fine.

2. **Static background**: Are the background, lighting, and camera angle \
consistent across all frames? There should be no noticeable camera movement, minimal lighting \
changes, and minimal background shifts.

3. **Subject in frame**: Is the main subject (animal/person) fully or mostly \
visible in every keyframe? FAIL if the subject exits, is cropped out, or becomes \
too small to be clearly seen in any frame.

3. **Physical plausibility**: Is the generated dynamics physically correct?

Respond in this exact format:
Motion: PASS or FAIL
Background: PASS or FAIL
Subject: PASS or FAIL
Physics: PASS or FAIL
Feedback: <one paragraph explaining any issues, or "None" if all pass>
"""


# ── Gemini helpers ───────────────────────────────────────────────────────────

def _gemini_client():
    """Create a Gemini client from GOOGLE_API_KEY."""
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def _call_gemini(client, contents):
    """Call Gemini with retry across model variants and rate-limit handling."""
    for model in GEMINI_TEXT_MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents
                )
                return (getattr(resp, "text", "") or "").strip()
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"    Rate limited on {model}, waiting {RATE_LIMIT_WAIT}s ...")
                    time.sleep(RATE_LIMIT_WAIT)
                else:
                    raise
        print(f"    Quota exhausted on {model}, trying next ...")
    raise RuntimeError("Gemini call failed: all models quota-exceeded.")


def _extract_prompt_from_plan(text):
    """Extract the Prompt: line from a planning response."""
    parts = re.split(r"\*{0,2}Prompt:\*{0,2}", text)
    return parts[-1].strip() if len(parts) > 1 else text


def _action_block(action_prompt):
    """Build the action description block for templates."""
    if action_prompt:
        return f'Action description: "{action_prompt}"\n'
    return "No action description was provided — the motion was predicted from the image.\n"


def _plan_prompt_gemini(action_prompt, image_path):
    """Ask Gemini to decompose the motion and write a video generation prompt.

    If action_prompt is provided, decomposes it into steps.
    If empty, Gemini predicts what happens next based on physics.
    """
    from PIL import Image
    client = _gemini_client()

    if action_prompt:
        template = PLAN_WITH_PROMPT_TEMPLATE.format(action_prompt=action_prompt)
    else:
        template = PLAN_NO_PROMPT_TEMPLATE

    pil_img = Image.open(image_path).convert("RGB")

    text = _call_gemini(client, [template, pil_img])
    prompt = _extract_prompt_from_plan(text)
    return prompt, text


def _refine_prompt_gemini(action_prompt, previous_prompt, feedback, image_path):
    """Ask Gemini to refine a video prompt based on quality check feedback."""
    from PIL import Image
    client = _gemini_client()
    template = REFINE_PROMPT_TEMPLATE.format(
        action_block=_action_block(action_prompt),
        previous_prompt=previous_prompt,
        feedback=feedback,
    )
    pil_img = Image.open(image_path).convert("RGB")

    text = _call_gemini(client, [template, pil_img])
    prompt = _extract_prompt_from_plan(text)
    return prompt, text


def _check_keyframes_gemini(action_prompt, states_dir, n_steps):
    """Ask Gemini to evaluate extracted keyframes for quality.

    Returns (passed: bool, feedback: str, full_text: str).
    """
    from PIL import Image
    client = _gemini_client()
    template = CHECK_KEYFRAMES_TEMPLATE.format(
        action_block=_action_block(action_prompt),
    )

    # Load all state images
    contents = [template]
    for i in range(n_steps + 1):
        img_path = os.path.join(states_dir, f"state_{i}.png")
        contents.append(Image.open(img_path).convert("RGB"))

    text = _call_gemini(client, contents)

    # Parse response
    motion_pass = bool(re.search(r"Motion:\s*PASS", text, re.IGNORECASE))
    bg_pass = bool(re.search(r"Background:\s*PASS", text, re.IGNORECASE))
    subject_pass = bool(re.search(r"Subject:\s*PASS", text, re.IGNORECASE))

    feedback_match = re.search(r"Feedback:\s*(.+)", text, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else text

    passed = motion_pass and bg_pass and subject_pass
    return passed, feedback, text


# ── Video generation backends ─────────────────────────────────────────────────

def _guess_mime(path):
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def _generate_video_veo(prompt, image_path, duration, save_path):
    """Generate video via Google Veo 3.1."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = types.Image(image_bytes=image_bytes, mime_type=_guess_mime(image_path))

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        image=image,
        config=types.GenerateVideosConfig(
            duration_seconds=duration,
        ),
    )

    while not operation.done:
        print("    Waiting for Veo generation ...")
        time.sleep(10)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Veo error: {operation.error}")

    video = operation.response.generated_videos[0]
    if hasattr(video.video, "video_bytes") and video.video.video_bytes:
        with open(save_path, "wb") as f:
            f.write(video.video.video_bytes)
    elif hasattr(video.video, "uri") and video.video.uri:
        import urllib.request
        uri = video.video.uri
        if api_key:
            sep = "&" if "?" in uri else "?"
            uri = f"{uri}{sep}key={api_key}"
        print("    Downloading video from URI ...")
        req = urllib.request.Request(uri)
        with urllib.request.urlopen(req) as resp, open(save_path, "wb") as f:
            f.write(resp.read())
    else:
        raise RuntimeError("Veo returned no video data or URI")


def _generate_video_kling(prompt, image_path, duration, save_path):
    """Generate video via Kling o3 (fal.ai). Supports arbitrary duration."""
    import fal_client
    import requests

    fal_key = os.environ.get("FAL_KEY", "").strip()
    if not fal_key:
        raise RuntimeError("Set FAL_KEY environment variable.")

    # Upload image to fal CDN
    image_url = fal_client.upload_file(image_path)

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"    [kling] {log['message']}")

    result = fal_client.subscribe(
        "fal-ai/kling-video/o3/standard/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "end_image_url": None,
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


VIDEO_BACKENDS = {
    "veo": _generate_video_veo,
    "kling": _generate_video_kling,
}


# ── Keyframe extraction ──────────────────────────────────────────────────────

def _get_frame_count(video_path):
    """Get total frame count from a video via ffprobe."""
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames",
         "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _extract_keyframes(video_path, image_path, states_dir, n_steps):
    """Extract N+1 evenly-spaced keyframes from the generated video.

    state_0 = original initial image (copy).
    state_1..N = evenly spaced from the video.
    """
    os.makedirs(states_dir, exist_ok=True)

    # state_0 = original image
    state_0 = os.path.join(states_dir, "state_0.png")
    if not os.path.isfile(state_0):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img.save(state_0)

    # Get total frames
    total_frames = _get_frame_count(video_path)
    if total_frames < n_steps:
        raise RuntimeError(
            f"Video has only {total_frames} frames, need at least {n_steps}"
        )

    # Extract frames at evenly-spaced positions (1/N, 2/N, ..., N/N)
    for i in range(1, n_steps + 1):
        state_path = os.path.join(states_dir, f"state_{i}.png")
        if os.path.isfile(state_path):
            continue

        frame_idx = round(i * (total_frames - 1) / n_steps)

        subprocess.run(
            [FFMPEG, "-i", video_path,
             "-vf", f"select=eq(n\\,{frame_idx})",
             "-frames:v", "1",
             state_path,
             "-y", "-loglevel", "error"],
            check=True,
        )
        print(f"    Extracted state_{i} from frame {frame_idx}/{total_frames - 1}")


# ── Per-instance processing ───────────────────────────────────────────────────

def _clear_states_and_video(states_dir, video_path, n_steps, attempt):
    """Rename failed video and remove state keyframes (except state_0) for re-generation."""
    if os.path.isfile(video_path):
        failed_path = video_path.replace(".mp4", f"_attempt{attempt}.mp4")
        os.rename(video_path, failed_path)
    for i in range(1, n_steps + 1):
        p = os.path.join(states_dir, f"state_{i}.png")
        if os.path.isfile(p):
            os.remove(p)


def process_instance(inst, data_root, output_root, video_model, duration,
                     n_steps, max_retries=MAX_REFLECTION_ATTEMPTS, gemini_plan=False):
    """Generate dynamics video and extract keyframes for one instance.

    Includes a reflection loop: after keyframe extraction, Gemini checks for
    coherent motion and static background. If either fails, Gemini refines the
    prompt and the video is regenerated (up to max_retries total attempts).
    """
    name = instance_name(inst)
    inst_dir = instance_output_dir(output_root, inst)
    states_dir = os.path.join(inst_dir, "states")
    video_path = os.path.join(inst_dir, "dynamics_static_cam.mp4")
    image_path = instance_image_path(data_root, inst)

    # Check if all keyframes already extracted AND passed quality check
    last_state = os.path.join(states_dir, f"state_{n_steps}.png")
    check_passed_file = os.path.join(inst_dir, "quality_check_passed.txt")
    skip_check = (max_retries <= 0)
    if os.path.isfile(last_state) and (skip_check or os.path.isfile(check_passed_file)):
        print(f"  [{name}] All state keyframes exist{'' if skip_check else ' and passed quality check'}, skipping.")
        return True

    if not os.path.isfile(image_path):
        print(f"  [{name}] WARNING: Image not found: {image_path}, skipping.")
        return False

    os.makedirs(inst_dir, exist_ok=True)

    # Center-crop to 16:9 and resize to 832x480 (HYWP native) — canonical input
    cropped_path = os.path.join(inst_dir, "init_16x9.png")
    if not os.path.isfile(cropped_path):
        _center_crop_and_resize(image_path, cropped_path)
    image_path = cropped_path

    action_prompt = construct_prompt(inst)
    generate_fn = VIDEO_BACKENDS[video_model]

    # Prompt files
    prompt_file = os.path.join(inst_dir, "video_prompt.txt")
    prompt_full_file = os.path.join(inst_dir, "motion_plan.txt")
    check_log_file = os.path.join(inst_dir, "quality_check_log.txt")

    # Load or generate initial prompt
    video_prompt = None
    if os.path.isfile(prompt_file):
        video_prompt = open(prompt_file).read().strip()
        print(f"  [{name}] Reusing cached prompt: {video_prompt[:100]}...")

    effective_retries = max(max_retries, 1)
    for attempt in range(1, effective_retries + 1):
        if not skip_check:
            print(f"  [{name}] Attempt {attempt}/{effective_retries}")

        # Step A: Write prompt (first attempt or refinement)
        if video_prompt is None:
            if not action_prompt and not gemini_plan:
                # No prompt provided — skip Gemini, use static camera prompt directly
                video_prompt = "Static camera."
                with open(prompt_file, "w") as f:
                    f.write(video_prompt)
                print(f"  [{name}] No prompt provided, using static camera prompt.")
            else:
                print(f"  [{name}] Planning motion via Gemini ...")
                try:
                    video_prompt, full_response = _plan_prompt_gemini(
                        action_prompt, image_path
                    )
                    with open(prompt_file, "w") as f:
                        f.write(video_prompt)
                    with open(prompt_full_file, "w") as f:
                        f.write(full_response)
                    print(f"  [{name}] Prompt: {video_prompt[:100]}...")
                except Exception as e:
                    print(f"  [{name}] ERROR: Motion planning failed: {e}")
                    video_prompt = action_prompt
                    print(f"  [{name}] Falling back to raw prompt.")

        # Step B: Generate video
        if not os.path.isfile(video_path):
            full_prompt = video_prompt + " " + VIDEO_PROMPT_SUFFIX
            print(f"  [{name}] Generating video ({video_model}, duration={duration}s) ...")
            print(f"    Prompt: {full_prompt[:120]}...")
            try:
                generate_fn(full_prompt, image_path, duration, video_path)
                print(f"  [{name}] Video saved: {video_path}")
            except Exception as e:
                print(f"  [{name}] ERROR: Video generation failed: {e}")
                return False
        else:
            print(f"  [{name}] Video exists: {video_path}")

        # Step C: Extract keyframes
        if not os.path.isfile(os.path.join(states_dir, f"state_{n_steps}.png")):
            print(f"  [{name}] Extracting {n_steps + 1} state keyframes ...")
            _extract_keyframes(video_path, image_path, states_dir, n_steps)

        # Step D: Quality check via Gemini (skip if max_retries=0)
        if skip_check:
            print(f"  [{name}] Done (quality check skipped).")
            return True

        print(f"  [{name}] Running quality check via Gemini ...")
        try:
            passed, feedback, full_check = _check_keyframes_gemini(
                action_prompt, states_dir, n_steps
            )
        except Exception as e:
            print(f"  [{name}] WARNING: Quality check failed: {e}")
            print(f"  [{name}] Proceeding without check.")
            # If check itself errors, accept the result
            with open(check_passed_file, "w") as f:
                f.write(f"check_error: {e}")
            return True

        # Log the check
        with open(check_log_file, "a") as f:
            f.write(f"=== Attempt {attempt} ===\n")
            f.write(f"Prompt: {video_prompt}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Full response:\n{full_check}\n\n")

        if passed:
            print(f"  [{name}] Quality check PASSED.")
            with open(check_passed_file, "w") as f:
                f.write(f"passed on attempt {attempt}")
            return True

        print(f"  [{name}] Quality check FAILED: {feedback[:200]}")

        if attempt < effective_retries:
            # Refine prompt and retry
            print(f"  [{name}] Refining prompt based on feedback ...")
            _clear_states_and_video(states_dir, video_path, n_steps, attempt)
            try:
                video_prompt, full_response = _refine_prompt_gemini(
                    action_prompt, video_prompt, feedback, image_path
                )
                with open(prompt_file, "w") as f:
                    f.write(video_prompt)
                with open(prompt_full_file, "w") as f:
                    f.write(full_response)
                print(f"  [{name}] Refined prompt: {video_prompt[:100]}...")
            except Exception as e:
                print(f"  [{name}] WARNING: Prompt refinement failed: {e}")
                print(f"  [{name}] Keeping current result.")
                with open(check_passed_file, "w") as f:
                    f.write(f"refinement_error on attempt {attempt}: {e}")
                return True

    # Exhausted all attempts — keep the last result
    print(f"  [{name}] Exhausted {effective_retries} attempts, keeping last result.")
    with open(check_passed_file, "w") as f:
        f.write(f"failed after {effective_retries} attempts")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instances_json", required=True,
                        help="Path to pipeline instances JSON")
    parser.add_argument("--data_root", default="",
                        help="Root for resolving relative image paths (default: cwd)")
    parser.add_argument("--output_root", required=True,
                        help="Output root directory")
    parser.add_argument("--video_model", choices=list(VIDEO_BACKENDS.keys()),
                        default="veo",
                        help="Video generation backend (default: veo)")
    parser.add_argument("--duration", type=int, default=5,
                        help="Video duration in seconds (default: 5)")
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help=f"Number of keyframe steps (default: {N_STEPS})")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Only process first N instances (for testing)")
    parser.add_argument("--max_retries", type=int,
                        default=MAX_REFLECTION_ATTEMPTS,
                        help=f"Max reflection attempts per instance (default: {MAX_REFLECTION_ATTEMPTS})")
    parser.add_argument("--skip_check", action="store_true",
                        help="Skip Gemini quality check (no reflection loop)")
    parser.add_argument("--gemini_plan", action="store_true",
                        help="Force Gemini motion planning even when no prompt is provided")
    args = parser.parse_args()

    with open(args.instances_json) as f:
        instances = json.load(f)

    if args.max_instances:
        instances = instances[:args.max_instances]

    print(f"Loaded {len(instances)} instances from {args.instances_json}")
    print(f"Video model: {args.video_model}")

    for idx, inst in enumerate(instances):
        name = instance_name(inst)
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(instances)}] {name}")
        print(f"{'='*60}")
        process_instance(inst, args.data_root, args.output_root,
                         args.video_model, args.duration, args.n_steps,
                         max_retries=0 if args.skip_check else args.max_retries,
                         gemini_plan=args.gemini_plan)

    print("\nAll instances complete.")


if __name__ == "__main__":
    main()
