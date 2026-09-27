#!/usr/bin/env python3
"""Shared utilities for the data engine pipeline.

Self-contained module with no external imports beyond stdlib.
Provides the poses-file loader, frame boundary computation, pose padding,
and standard instance helpers.
"""

import math
import os

# ── Constants ─────────────────────────────────────────────────────────────────
FRAMES_PER_LATENT = 4
DEFAULT_FPS = 16
N_STEPS = 3  # default number of pose segments / state keyframes

# ── Camera poses ─────────────────────────────────────────────────────────────
# Camera trajectories are supplied by the user in a poses JSON file:
#   {"train": {"<name>": "<pose string>", ...},
#    "test":  {"<name>": "<pose string>", ...}}
# A pose string is a comma-separated list of WASD segments "<action>-<latents>",
# with actions w/s/a/d (translate), up/down/left/right (rotate). Every pose must
# have the same number of segments (= n_steps). "train" trajectories are used to
# train DynaToken; "test" ones are held out for evaluation.


def load_poses(poses_json, max_poses=None, n_steps=None):
    """Return (poses, names) from a poses JSON file; names are "train/<n>" / "test/<n>".

    max_poses keeps only the first max_poses train poses (all test poses are kept).
    If n_steps is given, every pose must have exactly n_steps segments.
    """
    import json
    with open(poses_json) as f:
        spec = json.load(f)
    unknown = set(spec) - {"train", "test"}
    if unknown:
        raise ValueError(f"{poses_json}: unexpected keys {sorted(unknown)} (use 'train' / 'test')")
    train = list(spec.get("train", {}).items())
    test = list(spec.get("test", {}).items())
    if max_poses is not None:
        train = train[:max_poses]
    poses, names = [], []
    for split, items in (("train", train), ("test", test)):
        for name, pose in items:
            if n_steps is not None and len(parse_segments(pose)) != n_steps:
                raise ValueError(f"{poses_json}: pose {split}/{name} = {pose!r} "
                                 f"has {len(parse_segments(pose))} segments, expected n_steps={n_steps}")
            n_latents = 1 + sum(c for _, c in parse_segments(pose))
            if n_latents % 4:
                raise ValueError(f"{poses_json}: pose {split}/{name} = {pose!r} spans {n_latents} latents "
                                 f"(motion steps + 1); HY-WorldPlay needs a multiple of 4")
            poses.append(pose)
            names.append(f"{split}/{name}")
    if not poses:
        raise ValueError(f"{poses_json}: no poses")
    return poses, names


# ── Pose parsing ─────────────────────────────────────────────────────────────

def parse_segments(action_string):
    """Parse a pose string into a list of (action, count) tuples.

    >>> parse_segments("w-6, right-7, w-6")
    [('w', 6), ('right', 7), ('w', 6)]
    """
    segments = []
    for cmd in action_string.split(","):
        cmd = cmd.strip()
        if not cmd:
            continue
        action, count = cmd.rsplit("-", 1)
        segments.append((action, int(count)))
    return segments


def action_boundaries(action_string):
    """Return list of (action, end_frame) for each segment.

    >>> action_boundaries("w-6, right-7, w-6")
    [('w', 24), ('right', 52), ('w', 76)]
    """
    results = []
    frame = 0
    for action, count in parse_segments(action_string):
        frame += count * FRAMES_PER_LATENT
        results.append((action, frame))
    return results


def build_keyframes_json(action_string, fps=DEFAULT_FPS):
    """Build keyframe list from action string."""
    boundaries = action_boundaries(action_string)
    output = [
        {
            "frame": 0,
            "timestamp": _frame_to_ts(0, fps),
            "type": "initial",
            "action": None,
            "edit_prompt": None,
            "reference_frame": None,
            "reference_timestamp": None,
        }
    ]
    prev_frame = 0
    for action, frame in boundaries:
        output.append({
            "frame": frame,
            "timestamp": _frame_to_ts(frame, fps),
            "type": "action",
            "action": action,
            "edit_prompt": None,
            "reference_frame": prev_frame,
            "reference_timestamp": _frame_to_ts(prev_frame, fps),
        })
        prev_frame = frame
    return output


def pad_pose(pose_str, n_segments):
    """Build a padded partial pose for the first n_segments of a pose string.

    HYWP requires latent_num % 4 == 0. This function takes the first n_segments
    of the pose, then extends the last segment's count so total latents is
    a multiple of 4.

    Returns:
        (padded_pose_str, video_length, target_frame)
        - padded_pose_str: the padded pose string for HYWP
        - video_length: the corresponding video_length arg for HYWP
        - target_frame: the 0-indexed frame to extract (at the original boundary)

    >>> pad_pose("w-6, right-7, w-6", 1)
    ('w-7', 29, 24)
    >>> pad_pose("w-6, right-7, w-6", 2)
    ('w-6,right-9', 61, 52)
    >>> pad_pose("w-6, right-7, w-6", 3)
    ('w-6,right-7,w-6', 77, 76)
    """
    segments = parse_segments(pose_str)
    partial = segments[:n_segments]

    # Original boundary frame (before padding)
    total_motions = sum(c for _, c in partial)
    target_frame = total_motions * FRAMES_PER_LATENT

    # Pad to valid latent count
    latent_num = total_motions + 1  # +1 for initial latent
    padded_latent_num = math.ceil(latent_num / 4) * 4
    padding = padded_latent_num - latent_num

    if padding > 0:
        action, count = partial[-1]
        partial[-1] = (action, count + padding)

    padded_pose_str = ",".join(f"{a}-{c}" for a, c in partial)
    video_length = (padded_latent_num - 1) * FRAMES_PER_LATENT + 1

    return padded_pose_str, video_length, target_frame


# ── Standard instance helpers ────────────────────────────────────────────────
#
# Standard pipeline JSON format:
#   {
#     "image": "relative/path/to/image.png",   (relative to data_root)
#     "prompt": "action description",
#     "output": "output/subdir",               (relative to output_root)
#     "name": "short_label"                    (optional)
#   }

def instance_name(inst):
    """Get a short display name for an instance."""
    if "name" in inst:
        return inst["name"]
    # Derive from output path or image basename
    return os.path.basename(inst.get("output", "")) or \
           os.path.splitext(os.path.basename(inst["image"]))[0]


def instance_output_dir(output_root, inst):
    """Build output directory for an instance."""
    return os.path.join(output_root, inst["output"])


def instance_image_path(data_root, inst):
    """Resolve the image path for an instance.

    If inst["image"] is absolute, returns it directly.
    Otherwise, resolves relative to data_root.
    """
    img = inst["image"]
    if os.path.isabs(img):
        return img
    return os.path.join(data_root, img.lstrip("./"))


def construct_prompt(inst):
    """Get the text prompt for an instance, or empty string if not provided."""
    return inst.get("prompt", "") or ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _frame_to_ts(frame, fps):
    total_s = frame // fps
    return f"{total_s // 60:02d}:{total_s % 60:02d}"
