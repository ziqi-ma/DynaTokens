"""Validate the evaluation manifest and vendored release files using the stdlib.

Run: python3 metrics/verify_eval_manifest.py --manifest metrics/eval_manifest.json
The 16-frame released samples all satisfy the multiple-of-four latent constraint;
this does not establish the frame count of unreleased evaluation trajectories.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_bytes(path, expected_path):
    require(path == expected_path, f"Unexpected vendored file path: {path!r}")
    return (ROOT / path).read_bytes()


def check_matrix(matrix, size, label):
    require(isinstance(matrix, list) and len(matrix) == size, f"{label}: expected {size} rows")
    for row in matrix:
        require(isinstance(row, list) and len(row) == size, f"{label}: expected {size} columns")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in row),
                f"{label}: matrix entries must be finite numbers")


def validate_manifest(manifest_path):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(data["issue"] == "https://github.com/ziqi-ma/DynaTokens/issues/3", "Incorrect issue")
    require(data["paper"] == "https://arxiv.org/abs/2609.35704", "Incorrect paper")
    seen_scenes = set()
    scene_count = 0
    pose_count = 0

    def check_scenes(scenes, benchmark):
        nonlocal scene_count, pose_count
        require(isinstance(scenes, list), f"{benchmark}: scenes must be a list")
        for scene in scenes:
            scene_id = scene["id"]
            require(isinstance(scene_id, str) and re.fullmatch(r"[a-z0-9_]+", scene_id), "Invalid scene id")
            require(scene_id not in seen_scenes, f"Duplicate scene id: {scene_id}")
            seen_scenes.add(scene_id)
            require(scene["benchmark"] == benchmark, f"{scene_id}: incorrect benchmark")
            s3 = f"s3://dynatokens/dynatokens_checkpoints/{scene_id}/"
            for field, name in (("prompt_source", "prompt.txt"), ("init_image", "init_image.png"),
                                ("checkpoint", "diffusion_pytorch_model.safetensors")):
                require(scene[field] == s3 + name, f"{scene_id}: incorrect {field}")
            raw_prompt = file_bytes(scene["prompt_file"], f"metrics/eval_poses/{scene_id}/prompt.txt")
            require(hashlib.sha256(raw_prompt).hexdigest() == scene["prompt_sha256"], f"{scene_id}: prompt sha256 mismatch")
            if benchmark == "vbench2":
                require(isinstance(scene["prompt"], str) and bool(scene["prompt"].strip()), f"{scene_id}: missing prompt")
                require(scene["prompt_status"] == "released" and raw_prompt.decode("utf-8") == scene["prompt"],
                        f"{scene_id}: prompt differs from released file")
            else:
                require(scene["prompt"] is None and scene["prompt_status"] == "not_released_empty_prompt_txt" and not raw_prompt,
                        f"{scene_id}: prompt must record empty release file")
                require(isinstance(scene.get("prompt_note"), str) and bool(scene["prompt_note"].strip()), f"{scene_id}: missing prompt note")
            trajectories = scene["trajectories"]
            require(isinstance(trajectories, list) and bool(trajectories), f"{scene_id}: missing trajectories")
            seen_names = set()
            for trajectory in trajectories:
                name = trajectory["name"]
                require(isinstance(name, str) and re.fullmatch(r"[a-z0-9_]+", name), f"{scene_id}: invalid trajectory name")
                require(name not in seen_names, f"{scene_id}: duplicate trajectory {name}")
                seen_names.add(name)
                label = f"{scene_id}/{name}"
                require(trajectory["source"] == s3 + f"sample_poses/{name}/pose.json", f"{label}: incorrect source")
                require(trajectory["split"] == "unknown", f"{label}: split must be unknown")
                raw = file_bytes(trajectory["pose_json"], f"metrics/eval_poses/{scene_id}/{name}/pose.json")
                require(hashlib.sha256(raw).hexdigest() == trajectory["sha256"], f"{label}: sha256 mismatch")
                frames = json.loads(raw)
                require(isinstance(frames, dict) and bool(frames), f"{label}: missing frames")
                count = len(frames)
                require(set(frames) == {str(i) for i in range(count)}, f"{label}: non-consecutive frame keys")
                require(type(trajectory["frame_count"]) is int and trajectory["frame_count"] == count, f"{label}: frame count mismatch")
                require(count % 4 == 0, f"{label}: frame count is not a multiple of 4")
                for index, frame in frames.items():
                    require(isinstance(frame, dict), f"{label}/{index}: frame must be an object")
                    check_matrix(frame["w2c"], 4, f"{label}/{index}/w2c")
                    check_matrix(frame["intrinsic"], 3, f"{label}/{index}/intrinsic")
                pose_count += 1
            scene_count += 1

    for benchmark, target in (("vbench2", 20), ("worldscore", 15)):
        entry = data["benchmarks"][benchmark]
        require(type(entry["paper_target_scenes"]) is int and entry["paper_target_scenes"] == target, f"{benchmark}: incorrect paper target")
        require(type(entry["trajectories_per_scene"]) is int and entry["trajectories_per_scene"] == 6, f"{benchmark}: incorrect trajectory target")
        pending = entry["pending_maintainer_confirmation"]
        require(type(pending) is int and pending >= 0, f"{benchmark}: invalid pending count")
        scenes = entry["scenes"]
        require(isinstance(scenes, list) and len(scenes) + pending == target, f"{benchmark}: inconsistent scene counts")
        expected = {"paper_target": target, "verified": len(scenes), "pending": pending}
        require(data["coverage"][benchmark] == expected, f"{benchmark}: inconsistent coverage")
        check_scenes(scenes, benchmark)
    check_scenes(data["additional_released_scenes"]["scenes"], "physicsiq")
    return scene_count, pose_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        scene_count, pose_count = validate_manifest(args.manifest)
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {scene_count} scenes, {pose_count} trajectories; coverage, prompts, hashes and matrices verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
