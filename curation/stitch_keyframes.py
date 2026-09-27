#!/usr/bin/env python3
"""Stitch interpolated keyframe videos with original frames into a single MP4.

For each interp_{A:04d}_{B:04d}.mp4 in --videos-dir, the video is subsampled
to exactly B-A frames and placed at output positions A through B-1. All other
positions are filled from the original frames directory. The result is written
as a single MP4 at 16 fps.

Usage:
    python stitch_keyframes.py \
        --dir data/scene \
        --frames-dir data/scene/frames \
        [--videos-dir data/scene/videos] \
        [--output data/scene/stitched.mp4]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

FPS = 16


def _find_bin(name):
    env_override = os.environ.get(name.upper())
    if env_override and os.path.isfile(env_override):
        return env_override
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Cannot find {name}. Add it to PATH.")


def _extract_frames(mp4, tmp_dir, w, h):
    subprocess.run(
        [_find_bin("ffmpeg"), "-i", mp4,
         "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
         os.path.join(tmp_dir, "frame_%06d.png"), "-y"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(glob.glob(os.path.join(tmp_dir, "*.png")))


def _subsample(frames, n_target, trim_start=0, trim_end=0):
    if trim_start or trim_end:
        end = len(frames) - trim_end if trim_end else len(frames)
        trimmed = frames[trim_start:end]
        if trimmed:
            frames = trimmed
    n_src = len(frames)
    indices = np.round(np.linspace(0, n_src - 1, n_target)).astype(int)
    return [frames[i] for i in indices]


def _detect_frame_size(frames_dir, edited_dir=None):
    """Detect frame resolution from available images."""
    # Try frames_dir first
    for fname in sorted(os.listdir(frames_dir)) if os.path.isdir(frames_dir) else []:
        if fname.endswith(".png"):
            img = Image.open(os.path.join(frames_dir, fname))
            return img.size  # (w, h)
    # Fall back to edited_dir
    if edited_dir and os.path.isdir(edited_dir):
        for fname in sorted(os.listdir(edited_dir)):
            if fname.endswith(".png"):
                img = Image.open(os.path.join(edited_dir, fname))
                return img.size
    return None


def run(data_dir, frames_dir, videos_dir, output_path, trim=0):
    v0 = os.path.join(data_dir, "merged_keyframes_v0.json")
    initial = os.path.join(data_dir, "merged_keyframes_initial.json")
    input_path = v0 if os.path.isfile(v0) else initial

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No keyframes file found in {data_dir}")

    with open(input_path) as f:
        keyframes = json.load(f)

    total = max(kf["frame"] for kf in keyframes)
    print(f"Input : {input_path}")
    print(f"Total frames: {total}")

    # Detect frame size
    edited_dir = os.path.join(data_dir, "frames_edited")
    size = _detect_frame_size(frames_dir, edited_dir)
    if size is None:
        raise RuntimeError("Cannot detect frame size from frames_dir or frames_edited/")
    orig_w, orig_h = size
    print(f"Frame size: {orig_w}x{orig_h}")

    segments = []
    for mp4 in glob.glob(os.path.join(videos_dir, "interp_*.mp4")):
        stem = Path(mp4).stem
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        a, b = int(parts[1]), int(parts[2])
        segments.append((a, b, mp4))
    segments.sort()

    if not segments:
        raise FileNotFoundError(f"No interp_*.mp4 files found in {videos_dir}")

    covered = set()
    for a, b, _ in segments:
        for i in range(a, b):
            covered.add(i)

    with tempfile.TemporaryDirectory() as out_frames_dir:

        def out_frame(i):
            return os.path.join(out_frames_dir, f"frame_{i + 1:04d}.png")

        def orig_frame(i):
            return os.path.join(frames_dir, f"frame_{i + 1:04d}.png")

        n_seg = len(segments)
        for idx, (a, b, mp4) in enumerate(segments):
            n_target = b - a
            trim_start = 0 if idx == 0 else trim
            trim_end = 0 if idx == n_seg - 1 else trim
            print(f"  {Path(mp4).name}: subsampling to {n_target} frames ...")
            with tempfile.TemporaryDirectory() as tmp:
                all_frames = _extract_frames(mp4, tmp, orig_w, orig_h)
                print(f"    extracted {len(all_frames)}, trim_start={trim_start} trim_end={trim_end}, subsampling to {n_target}")
                selected = _subsample(all_frames, n_target, trim_start=trim_start, trim_end=trim_end)
                for out_i, src in enumerate(selected):
                    shutil.copy(src, out_frame(a + out_i))

        uncovered = [i for i in range(total) if i not in covered]
        print(f"  Copying {len(uncovered)} original frames ...")

        def _copy_orig(i):
            src = orig_frame(i)
            if os.path.isfile(src):
                shutil.copy(src, out_frame(i))

        with ThreadPoolExecutor() as pool:
            pool.map(_copy_orig, uncovered)

        print(f"\nAll {total} frames assembled. Writing {output_path} ...")
        import imageio
        frame_paths = sorted(glob.glob(os.path.join(out_frames_dir, "frame_*.png")))
        writer = imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=8)
        for fp in frame_paths:
            writer.append_data(np.array(Image.open(fp)))
        writer.close()
        print(f"Done -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", required=True, help="Data directory")
    parser.add_argument("--frames-dir", required=True, help="Original frames directory")
    parser.add_argument("--videos-dir", default=None, help="Interpolated videos directory (default: videos/ inside --dir)")
    parser.add_argument("--output", default=None, help="Output MP4 path (default: stitched.mp4 inside --dir)")
    parser.add_argument("--trim", type=int, default=0, help="Frames to trim at chunk boundaries (default: 0)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.dir)
    videos_dir = args.videos_dir or os.path.join(data_dir, "videos")
    output_path = args.output or os.path.join(data_dir, "stitched.mp4")
    run(data_dir, args.frames_dir, videos_dir, output_path, trim=args.trim)


if __name__ == "__main__":
    main()
