"""Generate camera poses without torch and validate them against released poses.

The full dynatokens/make_pose_json.py needs the DynaToken environment (torch and
hyvideo). This torch-free equivalent uses the same trajectory math and can be
validated against the released VBench2 files:

  python metrics/generate_eval_poses.py verify --manifest metrics/eval_manifest.json
  python metrics/generate_eval_poses.py generate --pose 'right-7,up-8' --out-dir /tmp/pose

Folder-name pose strings in the manifest are inferred and checked numerically;
the release does not itself provide those strings.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model/hywp/hyvideo"))
from generate_custom_trajectory import generate_camera_trajectory_local


def parse_pose_string(pose_string):
    """Use generate.py's per-frame motion values with integer latent counts."""
    speed = 0.08
    angle = np.deg2rad(3)
    actions = {
        "w": ("forward", speed), "s": ("forward", -speed),
        "a": ("right", -speed), "d": ("right", speed),
        "up": ("pitch", angle), "down": ("pitch", -angle),
        "left": ("yaw", -angle), "right": ("yaw", angle),
    }
    motions = []
    for command in pose_string.split(","):
        match = re.fullmatch(r"\s*([a-z]+)\s*-\s*([0-9]+)\s*", command)
        if not match or match.group(1) not in actions:
            raise ValueError(f"Invalid pose command: {command!r}")
        field, amount = actions[match.group(1)]
        motions.extend({field: amount} for _ in range(int(match.group(2))))
    if (len(motions) + 1) % 4:
        raise ValueError("Motion steps plus the initial frame must be a multiple of 4")
    return motions


def pose_string_to_pose_json(pose_string):
    raw_intrinsic = np.array([
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ])
    intrinsic = raw_intrinsic.copy()
    intrinsic[0, 0] /= raw_intrinsic[0, 2] * 2
    intrinsic[1, 1] /= raw_intrinsic[1, 2] * 2
    intrinsic[0, 2] = intrinsic[1, 2] = 0.5
    poses = generate_camera_trajectory_local(parse_pose_string(pose_string))
    return {
        str(i): {"w2c": np.linalg.inv(c2w).tolist(), "intrinsic": intrinsic.tolist()}
        for i, c2w in enumerate(poses)
    }


def write_pose(pose_string, out_dir):
    pose = pose_string_to_pose_json(pose_string)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = out_dir / "pose.json"
    with pose_path.open("w", encoding="utf-8") as stream:
        json.dump(pose, stream, indent=2)
    (out_dir / "pose_string.txt").write_text(pose_string + "\n", encoding="utf-8")
    return pose_path


def verify_release(manifest):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    checked = 0
    max_error = 0.0
    for scene in data["benchmarks"]["vbench2"]["scenes"]:
        for trajectory in scene["trajectories"]:
            if "pose_string" not in trajectory:
                continue
            actual = pose_string_to_pose_json(trajectory["pose_string"])
            released = json.loads((ROOT / trajectory["pose_json"]).read_text(encoding="utf-8"))
            if set(actual) != set(released):
                raise ValueError(f"Frame keys differ for {scene['id']}/{trajectory['name']}")
            for frame in actual:
                for field in ("w2c", "intrinsic"):
                    a = np.asarray(actual[frame][field], dtype=float)
                    b = np.asarray(released[frame][field], dtype=float)
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError(f"Invalid {field} in {scene['id']}/{trajectory['name']}")
                    error = float(np.max(np.abs(a - b)))
                    max_error = max(max_error, error)
                    if error > 1e-6:
                        raise ValueError(f"{scene['id']}/{trajectory['name']} {field} differs by {error:g}")
            checked += 1
    if not checked:
        raise ValueError("No VBench2 pose strings available to verify")
    print(f"PASS: reproduced {checked} released VBench2 poses; maximum absolute matrix error {max_error:.3g} (tolerance 1e-6)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--pose", required=True)
    generate.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify_release(args.manifest)
        else:
            print(f"{args.pose!r} -> {write_pose(args.pose, args.out_dir)}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
