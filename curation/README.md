# Data Curation

We curate a few supervision videos per scene at test time. Gemini plans coarse motion, a video generation model (Kling or Veo) generates the scene dynamics, and the base camera-controlled model renders the camera motion. We combine them because video generation models produce good object dynamics but cannot follow a camera path, while the camera-controlled model follows the camera path but struggles with dynamics.

## How It Works

Given an initial frame (and optionally a text prompt), and camera trajectories of `N_STEPS` segments each:

### Step 1: Motion Planning + Video Generation (`generate_states.py`)

Gemini acts as a motion planner. With a text prompt, it decomposes the described action into keyframes from the initial to the final state; without one, it predicts the most natural continuation from physical cues in the image. It then writes a video generation prompt describing that progression, and the video model generates a static-camera video from it. `N_STEPS + 1` evenly spaced state keyframes are extracted: `state_0` (the initial image) to `state_N`.

Reflection loop: Gemini checks the keyframes for (1) coherent motion and (2) a static background. If either fails, it revises the plan and the video is regenerated, up to 3 attempts (`--max_retries`; `--skip_check` disables the check).

### Step 2: De-blurring (`deblur_keyframes.py`)

Gemini checks each state `state_1 … state_N` for motion blur; blurred ones are sharpened with Gemini image editing and saved as `state_i_deblurred.png`, which Step 3 uses instead of `state_i.png`.

### Step 3: Camera Re-rendering (`hywp_rerender.py`)

For each camera trajectory in `POSES_JSON` and each state `i`, the base camera-controlled model renders `state_i` along the first `i` segments of the trajectory with a neutral prompt ("A static scene."), so it moves the camera without adding dynamics. This gives `(state_i, view_i)`. Work is sharded across `NUM_GPUS` GPUs.

### Steps 4-5: Assembly, Interpolation, Stitching, Review (`assemble_keyframes.py`)

For each trajectory, the pipeline places the keyframes `(state_i, view_i)` at the segment boundaries, interpolates between consecutive keyframes with Kling (Gemini writes each transition prompt from the two images and the camera action), and stitches the clips into `stitched.mp4` at 16 fps. Finally Gemini reviews every `stitched.mp4` against the prompt (text alignment, object artifacts) and saves the verdict in `stitched_review.json`.

## Instance JSON Format

The pipeline uses a standardized JSON format:

```json
[
    {
        "image": "images/scene_a.png",
        "prompt": "<action description>",
        "output": "scene_a"
    },
    {
        "image": "images/scene_b.jpg",
        "output": "scene_b"
    }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `image` | yes | Path to initial image (absolute, or relative to `DATA_ROOT`). Supports PNG, JPEG, etc. |
| `prompt` | no | Action description. If omitted, Gemini predicts what happens next based on physics. |
| `output` | yes | Output subdirectory, relative to `OUTPUT_ROOT` |
| `name` | no | Display name (derived from `output` basename if missing) |

Write this file by hand, or generate it from a benchmark's metadata with a converter: `convert_worldscore.py` (WorldScore dynamic JSON) and `convert_physicsiq.py` (Physics-IQ `descriptions.csv`; drops its "Static shot with no camera movement." sentence).

## Quick Start

Run from the repository root with the `dynatoken` environment active, and export your API
keys first (they are never stored in the repository):

```bash
export GOOGLE_API_KEY=...   # Gemini (+ Veo)
export FAL_KEY=...          # Kling via fal.ai

INSTANCES_JSON=my_scenes.json POSES_JSON=my_poses.json OUTPUT_ROOT=outputs/curation/my_scenes \
    N_STEPS=2 bash curation/run.sh

# quick test: 1 scene, 2 train poses
MAX_INSTANCES=1 MAX_POSES=2 INSTANCES_JSON=... POSES_JSON=... OUTPUT_ROOT=... bash curation/run.sh
```

Converters that build the scenes JSON from a benchmark:

```bash
python curation/convert_worldscore.py --dynamic_json <dynamic_test.json> --dataset_root <WorldScore-Dataset> --output_json scenes.json
python curation/convert_physicsiq.py --descriptions_csv <descriptions.csv> --switch_frames_dir <switch-frames> --output_json scenes.json
```

Each stage can also be run on its own (useful for debugging or re-running a single step):

```bash
export INSTANCES_JSON=... POSES_JSON=... OUTPUT_ROOT=...
source curation/pipeline/defaults.sh

bash curation/pipeline/generate_states.sh     # Step 1: motion planning + video gen (API)
bash curation/pipeline/deblur.sh              # Step 2: state-image deblur (API)
bash curation/pipeline/hywp_rerender.sh       # Step 3: camera re-rendering (GPU)
bash curation/pipeline/assemble.sh            # Steps 4-5: interpolation + stitch (API)
```

## Configuration

All settings are environment variables, with defaults in `pipeline/defaults.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `INSTANCES_JSON` | required | Scenes JSON, see [Instance JSON Format](#instance-json-format) |
| `POSES_JSON` | required | Camera trajectories, see [Camera Poses](#camera-poses) |
| `OUTPUT_ROOT` | required | Output root; each scene goes to `$OUTPUT_ROOT/<output>` |
| `N_STEPS` | `3` | Segments per camera pose (= state keyframes); must match `POSES_JSON` |
| `DATA_ROOT` | `datasets/` | Root for resolving relative image paths |
| `VIDEO_MODEL` | `kling` | Video backend: `kling` or `veo` |
| `DURATION` | `5` | Video duration in seconds (Kling snaps to 5 or 10) |
| `NUM_GPUS` / `GPU_START` | `4` / `0` | GPUs for camera re-rendering |
| `HYWP_STEPS` | `30` | Denoising steps for camera re-rendering |
| `MAX_POSES` | (all) | Limit to first N train poses (test poses always included) |
| `MAX_INSTANCES` | (all) | Limit to first N scenes |
| `MODEL_PATH` / `ACTION_CKPT` | auto (HF cache) | Base model checkpoints |
| `API_CONDA` / `HYWP_CONDA` | (unset) | Optional conda envs for the API / GPU stages; default: current env |
| `GOOGLE_API_KEY` | required | Gemini (motion planning, quality check, interpolation prompts) + Veo |
| `FAL_KEY` | required | Kling via fal.ai (video generation + interpolation) |

## Output Structure

```
{OUTPUT_ROOT}/{instance_output}/
|-- init_16x9.png                       # Center-cropped + resized to 832x480
|-- video_prompt.txt                    # Final video generation prompt
|-- motion_plan.txt                     # Full Gemini reasoning (keyframes + prompt)
|-- dynamics_static_cam.mp4             # Static-camera dynamics video (final attempt)
|-- dynamics_static_cam_attempt1.mp4    # Failed attempt video (if reflection loop retried)
|-- quality_check_passed.txt            # Status: "passed on attempt N" or "failed after N attempts"
|-- quality_check_log.txt               # Full log of all quality check attempts
|-- states/
|   |-- state_0.png                     # = init_16x9.png (initial frame)
|   |-- state_1.png ... state_N.png     # Extracted evenly from the video (state_N = last frame)
|   |-- state_i_deblurred.png           # Step 2 output, only for states detected as blurred
|-- train/
|   |-- {pose_name}/
|       |-- hywp_rerender/
|       |   |-- state_i_view.png        # State i rendered at camera view i
|       |   |-- state_i_gen.mp4         # Full camera re-rendering video
|       |-- merged_keyframes_initial.json  # Keyframe metadata for interpolation
|       |-- frames_edited/              # Assembled keyframes (edited_0000.png, ...)
|       |-- videos/                     # Kling interpolated clips (interp_0000_0024.mp4, ...)
|       |-- stitched.mp4                # Final supervision video
|       |-- stitched_review.json        # Gemini review ({"passed": ...})
|-- test/
    |-- ...                             # Same structure as train/
```

## File Structure

```
curation/
|-- run.sh                       # Entry point: all stages for a scenes JSON
|-- generate_states.py           # Step 1: Gemini motion planning + video gen + quality check
|-- deblur_keyframes.py          # Step 2: de-blur extracted state images
|-- hywp_rerender.py             # Step 3: camera re-rendering (multi-GPU)
|-- assemble_keyframes.py        # Steps 4-5: assemble keyframes + run interpolation + stitch
|-- interpolate_keyframes.py     # Kling interpolation between consecutive keyframe pairs
|-- stitch_keyframes.py          # Subsample interpolated clips + assemble into single MP4
|-- keyframe_utils.py            # Shared: poses file, frame boundaries, padding, instance helpers
|-- convert_worldscore.py        # WorldScore JSON -> scenes JSON
|-- convert_physicsiq.py         # Physics-IQ descriptions.csv -> scenes JSON
|
|-- pipeline/                    # Stage scripts
    |-- defaults.sh              # Shared defaults (VIDEO_MODEL, NUM_GPUS, checkpoints, optional conda envs)
    |-- generate_states.sh       # Stage 1 (API)
    |-- deblur.sh                # Stage 2 (API)
    |-- hywp_rerender.sh         # Stage 3 (GPU, multi-GPU launch)
    |-- assemble.sh              # Stages 4-5 (API)
    |-- run_all.sh               # All stages in order
```

## Camera Poses

The camera trajectories are an input: `POSES_JSON` points to a file

```json
{
  "train": {"<name>": "<pose string>", ...},
  "test":  {"<name>": "<pose string>", ...}
}
```

Each trajectory is rendered into `<scene>/<train|test>/<name>/`. `train` trajectories are used to
train DynaToken; `test` ones are new camera paths to render.

- Pose string: comma-separated segments `<action>-<number-of-latents>`
- Translation: `w` (forward), `s` (backward), `a` (strafe left), `d` (strafe right)
- Rotation: `up` (tilt up), `down` (tilt down), `left` (pan left), `right` (pan right)
- Constraints: every pose has `N_STEPS` segments (one state keyframe per segment) and the same length; motion steps + 1 must be a multiple of 4
- Filtering: `MAX_POSES=N` limits to the first N train poses; all test poses are always included

## Pose Padding

The base model needs the number of latents to be a multiple of 4. When rendering the first `i` segments of a trajectory (Step 3), the last of those segments is extended to satisfy this, and the frame at the original segment boundary is extracted, so the padding does not change the keyframe.

## Environment

All stages run in the repository's `dynatoken` environment (`environment.yml` at the repo root). Set `API_CONDA` / `HYWP_CONDA` only if you prefer separate conda envs for the API stages and the GPU stage.

## Quality Filtering

DynaTokens training is robust and works without filtering for many scenes. For better quality, `dynatokens/prepare_training_data.py` by default uses only the training videos whose `stitched_review.json` passed the Gemini review. Use `--passing-list <file>` (one video path per line) to choose videos yourself, or `--no-review-filter` to use all of them.
