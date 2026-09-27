#!/usr/bin/env python3
"""Evaluate how well generated videos follow their intended camera trajectories (ViPE).

For each gen.mp4 under
    <runs_root>/<run>/eval/ckpt<ckpt>/{seen,unseen}_<traj>/gen.mp4
(the layout written by dynatokens/scripts_*/eval_batch.sh; use --flat-layout
for <runs_root>/<run>/{seen,unseen}_<traj>/gen.mp4)
this driver:
  1. Reads the intended pose string from <pose dir>/pose_string.txt (or --poses-json).
  2. Computes per-segment frame boundaries using FRAMES_PER_LATENT=4.
  3. Runs NVIDIA ViPE on the mp4 to estimate per-frame camera poses.
  4. Extracts the 4x4 c2w matrices at each segment's before- and after-frame,
     computes the relative motion delta inv(T_b) @ T_e, and reports translation
     + rotation in axis-angle form.
  5. Optionally reconstructs the expected HYWP ground-truth pose delta from
     HYWP's motion constants (forward 0.08/latent, yaw/pitch 3 deg/latent)
     by re-implementing generate_camera_trajectory_local locally.
  6. Writes one trajectory_eval.json per video plus an aggregate summary per
     (run, ckpt).

This module must run inside the `vipe` conda env. HYWP code is NOT imported;
we re-derive the GT trajectory from the same constants in a local helper so
the metric has no runtime dependency on HYWP.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep the script dir off sys.path so nothing there shadows the installed `vipe` package.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p != _SCRIPT_DIR and p != ""]

# Reuse action_boundaries from the curation pipeline.
# keyframe_utils.py only imports stdlib, so this is safe across conda envs.
sys.path.insert(0, str(REPO_ROOT / "curation"))
from keyframe_utils import (  # noqa: E402
    FRAMES_PER_LATENT,
    action_boundaries,
    parse_segments,
)

# Optional poses JSON ({"train": {name: pose}, "test": {name: pose}}), set by --poses-json;
# only needed for pose dirs without a pose_string.txt.
POSES_JSON: Path | None = None

# ── HYWP motion constants (kept in sync with model/hywp/hyvideo/generate.py) ──
# These are re-stated here (not imported) so this script has no HYWP dep.
_FORWARD_SPEED = 0.08              # units per motion (1 motion = 1 latent = 4 video frames)
_YAW_SPEED_RAD = np.deg2rad(3.0)   # rad per motion
_PITCH_SPEED_RAD = np.deg2rad(3.0)

logger = logging.getLogger("eval_camera_trajectory")


# ─────────────────────────────────────────────────────────────────────────────
# GT trajectory reconstruction (local copy of HYWP's generate_camera_trajectory_local)
# ─────────────────────────────────────────────────────────────────────────────
def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _action_to_motion(action: str) -> dict:
    """Map one pose-string action token to HYWP's per-motion delta dict."""
    if action == "w":
        return {"forward": _FORWARD_SPEED}
    if action == "s":
        return {"forward": -_FORWARD_SPEED}
    if action == "a":
        return {"right": -_FORWARD_SPEED}
    if action == "d":
        return {"right": _FORWARD_SPEED}
    if action == "up":
        return {"pitch": _PITCH_SPEED_RAD}
    if action == "down":
        return {"pitch": -_PITCH_SPEED_RAD}
    if action == "left":
        return {"yaw": -_YAW_SPEED_RAD}
    if action == "right":
        return {"yaw": _YAW_SPEED_RAD}
    raise ValueError(f"Unknown action: {action!r}")


def expected_poses(pose_string: str) -> list[np.ndarray]:
    """Return one 4x4 c2w matrix per latent index (len = total_latents + 1).

    This mirrors HYWP's generate_camera_trajectory_local but is re-derived
    locally to avoid a runtime dep on HYWP. The convention is HYWP-internal:
      forward = +Z cam-local, right = +X cam-local, up = +Y cam-local.
    Video frame k*FRAMES_PER_LATENT corresponds to latent k here.
    """
    segments = parse_segments(pose_string)
    motions: list[dict] = []
    for action, count in segments:
        motion = _action_to_motion(action)
        for _ in range(count):
            motions.append(motion)

    poses: list[np.ndarray] = [np.eye(4)]
    T = np.eye(4)
    for move in motions:
        if "yaw" in move:
            T[:3, :3] = T[:3, :3] @ _rot_y(move["yaw"])
        if "pitch" in move:
            T[:3, :3] = T[:3, :3] @ _rot_x(move["pitch"])
        forward = move.get("forward", 0.0)
        if forward:
            T[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, forward])
        right = move.get("right", 0.0)
        if right:
            T[:3, 3] += T[:3, :3] @ np.array([right, 0.0, 0.0])
        poses.append(T.copy())
    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Pose delta decomposition
# ─────────────────────────────────────────────────────────────────────────────
def _rotation_matrix_to_axis_angle_deg(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (rotvec_deg (3,), angle_deg). rotvec = axis * angle."""
    # scipy.spatial.transform.Rotation.from_matrix handles numerical noise.
    from scipy.spatial.transform import Rotation as ScipyRotation
    rotvec_rad = ScipyRotation.from_matrix(R).as_rotvec()
    angle_rad = float(np.linalg.norm(rotvec_rad))
    return np.rad2deg(rotvec_rad), float(np.rad2deg(angle_rad))


def compute_delta(T_before: np.ndarray, T_after: np.ndarray) -> dict:
    """Pose of the after-camera in the before-camera's local frame.

    For c2w matrices T, the relative motion is inv(T_before) @ T_after. The
    translation component describes where the after-camera's origin sits in
    the before-camera's coordinate system.
    """
    T_rel = np.linalg.inv(T_before) @ T_after
    t = T_rel[:3, 3].astype(float)
    rotvec_deg, angle_deg = _rotation_matrix_to_axis_angle_deg(T_rel[:3, :3])
    return {
        "translation": t.tolist(),
        "translation_magnitude": float(np.linalg.norm(t)),
        "rotation_rotvec_deg": rotvec_deg.tolist(),
        "rotation_angle_deg": angle_deg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory metadata lookup
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrajectorySpec:
    split: str          # "seen" | "unseen"
    trajectory: str     # e.g. "w_d"
    pose_string: str    # e.g. "w-7, d-8"
    segments: list      # [(action, start_frame, end_frame), ...]
    total_frames: int


def _lookup_pose_string(pose_dir_name: str, pose_dir: Path | None = None) -> tuple[str, str, str]:
    """Given e.g. 'seen_w_d' return (split, traj, pose_string).

    The pose string is read from <pose_dir>/pose_string.txt (written by
    dynatokens/make_inference_jobs.py), else looked up in --poses-json
    ("seen" -> "train", "unseen" -> "test").
    """
    if pose_dir_name.startswith("seen_"):
        split, traj, key = "seen", pose_dir_name[len("seen_"):], "train"
    elif pose_dir_name.startswith("unseen_"):
        split, traj, key = "unseen", pose_dir_name[len("unseen_"):], "test"
    else:
        raise ValueError(
            f"Unexpected pose dir name {pose_dir_name!r} (needs seen_/unseen_ prefix)"
        )

    if pose_dir is not None and (pose_dir / "pose_string.txt").is_file():
        return split, traj, (pose_dir / "pose_string.txt").read_text().strip()
    if POSES_JSON is not None:
        poses = json.loads(Path(POSES_JSON).read_text())
        for k in (key, "test" if key == "train" else "train"):
            if traj in poses.get(k, {}):
                return split, traj, poses[k][traj]
    raise KeyError(f"No pose string for {pose_dir_name!r}: no pose_string.txt in {pose_dir} "
                   f"and not found in --poses-json ({POSES_JSON})")


def build_trajectory_spec(pose_dir_name: str, pose_dir: Path | None = None) -> TrajectorySpec:
    split, traj, pose_string = _lookup_pose_string(pose_dir_name, pose_dir)
    boundaries = action_boundaries(pose_string)   # [(action, end_frame), ...]
    segments = []
    prev_end = 0
    for action, end in boundaries:
        segments.append((action, prev_end, end))
        prev_end = end
    total_frames = boundaries[-1][1] + 1          # e.g. 60 -> 61 frames
    return TrajectorySpec(
        split=split, trajectory=traj, pose_string=pose_string,
        segments=segments, total_frames=total_frames,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ViPE integration
# ─────────────────────────────────────────────────────────────────────────────
def make_vipe_config(pipeline: str, output_path: Path):
    """Build the hydra config the ViPE pipeline expects.

    Lazy-imports hydra/omegaconf so that `--help` works without the vipe env.
    """
    import hydra                                             # noqa: F401
    from vipe import get_config_path                         # noqa: F401

    overrides = [
        f"pipeline={pipeline}",
        f"pipeline.output.path={output_path}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
    ]
    # Use initialize_config_dir + compose (same pattern as vipe/cli/main.py).
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)
    return args


def run_vipe_on_video(video_path: Path, vipe_output_dir: Path, pipeline: str) -> Path:
    """Run ViPE on a single mp4. Returns path to the pose npz.

    The pipeline is rebuilt per video (matching vipe/run.py) because models
    hold per-video state internally. Model weights are cached, so rebuild
    cost is limited to constructor overhead (not file reads).
    """
    from vipe import make_pipeline
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    args = make_vipe_config(pipeline, vipe_output_dir)
    vipe_output_dir.mkdir(parents=True, exist_ok=True)

    pose_npz = vipe_output_dir / "pose" / f"{video_path.stem}.npz"
    if pose_npz.exists():
        return pose_npz

    stream = ProcessedVideoStream(RawMp4Stream(video_path), []).cache(desc=f"Reading {video_path.name}")
    make_pipeline(args.pipeline).run(stream)

    if not pose_npz.exists():
        raise FileNotFoundError(
            f"ViPE finished but produced no pose file at {pose_npz}. "
            "Likely SLAM diverged or the clip was too short."
        )
    return pose_npz


def load_vipe_poses(pose_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (inds: (N,), c2w: (N, 4, 4)) in OpenCV convention."""
    with np.load(pose_npz) as data:
        inds = np.asarray(data["inds"]).astype(int)
        c2w = np.asarray(data["data"]).astype(float)
    return inds, c2w


def load_vipe_intrinsics(intr_npz: Path) -> dict | None:
    if not intr_npz.exists():
        return None
    with np.load(intr_npz) as data:
        return {
            "inds": np.asarray(data["inds"]).astype(int).tolist(),
            "data": np.asarray(data["data"]).astype(float).tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-video evaluation
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_available_frame(inds: np.ndarray, target: int) -> int:
    """Given the set of available frame indices, return the one closest to target."""
    if len(inds) == 0:
        raise ValueError("ViPE returned no pose frames")
    idx = int(np.argmin(np.abs(inds - target)))
    return int(inds[idx])


def evaluate_video(
    video_path: Path,
    pose_dir_name: str,
    pipeline: str,
    include_gt: bool,
) -> dict:
    spec = build_trajectory_spec(pose_dir_name, video_path.parent)
    vipe_out = video_path.parent / "vipe_results"
    pose_npz = run_vipe_on_video(video_path, vipe_out, pipeline=pipeline)
    inds, c2w = load_vipe_poses(pose_npz)
    ind_to_row = {int(i): int(r) for r, i in enumerate(inds)}

    gt_poses = expected_poses(spec.pose_string) if include_gt else None

    segments_out = []
    for action, start_frame, end_frame in spec.segments:
        actual_start = _nearest_available_frame(inds, start_frame)
        actual_end = _nearest_available_frame(inds, end_frame)
        T_b = c2w[ind_to_row[actual_start]]
        T_e = c2w[ind_to_row[actual_end]]
        seg = {
            "action": action,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "actual_start_frame": actual_start,
            "actual_end_frame": actual_end,
            "delta_estimated": compute_delta(T_b, T_e),
        }
        if gt_poses is not None:
            # Latent index = frame // FRAMES_PER_LATENT.
            lat_b = start_frame // FRAMES_PER_LATENT
            lat_e = end_frame // FRAMES_PER_LATENT
            seg["delta_expected"] = compute_delta(gt_poses[lat_b], gt_poses[lat_e])
        segments_out.append(seg)

    intr_npz = vipe_out / "intrinsics" / f"{video_path.stem}.npz"
    return {
        "video": str(video_path.name),
        "video_path": str(video_path),
        "trajectory": spec.trajectory,
        "split": spec.split,
        "pose_string": spec.pose_string,
        "fps": 16,
        "total_frames": spec.total_frames,
        "segments": segments_out,
        "per_frame": {
            "inds": inds.astype(int).tolist(),
            "c2w": c2w.astype(float).tolist(),
            "convention": "opencv_c2w",
        },
        "intrinsics": load_vipe_intrinsics(intr_npz),
        "vipe_pipeline": pipeline,
        "vipe_raw_dir": str(vipe_out.relative_to(video_path.parent)),
        "include_gt": bool(include_gt),
        "gt_convention": "hywp_internal (forward=+Z, right=+X)" if include_gt else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_ckpt_dir(hywp_runs_root: Path, run: str, ckpt: int, flat: bool = False) -> Path:
    if flat:
        ckpt_dir = hywp_runs_root / run
    else:
        ckpt_dir = hywp_runs_root / run / "eval" / f"ckpt{ckpt}"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt dir does not exist: {ckpt_dir}")
    return ckpt_dir


def _discover_pose_dirs(ckpt_dir: Path, filter_poses: Iterable[str] | None) -> list[Path]:
    all_dirs = sorted(
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and (p.name.startswith("seen_") or p.name.startswith("unseen_"))
    )
    if filter_poses is None:
        return all_dirs
    wanted = set(filter_poses)
    selected = [p for p in all_dirs if p.name in wanted]
    missing = wanted - {p.name for p in selected}
    if missing:
        raise FileNotFoundError(
            f"--poses asked for {sorted(missing)} but those dirs are not under {ckpt_dir}"
        )
    return selected


def _infer_instance_from_run(run: str) -> str:
    """Instance name is typically the run dirname with the trailing _<suffix> stripped."""
    parts = run.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else run


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _summarize(video_result: dict) -> dict:
    return {
        "trajectory": video_result["trajectory"],
        "split": video_result["split"],
        "pose_string": video_result["pose_string"],
        "total_frames": video_result["total_frames"],
        "segments": [
            {
                "action": s["action"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "actual_start_frame": s["actual_start_frame"],
                "actual_end_frame": s["actual_end_frame"],
                "estimated_translation_magnitude": s["delta_estimated"]["translation_magnitude"],
                "estimated_rotation_angle_deg": s["delta_estimated"]["rotation_angle_deg"],
                **({
                    "expected_translation_magnitude": s["delta_expected"]["translation_magnitude"],
                    "expected_rotation_angle_deg": s["delta_expected"]["rotation_angle_deg"],
                } if "delta_expected" in s else {}),
            }
            for s in video_result["segments"]
        ],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hywp-runs-root", type=Path,
                        default=REPO_ROOT / "outputs" / "runs",
                        help="Root dir containing per-run eval folders (e.g. outputs/runs/vbench).")
    parser.add_argument("--run", required=True,
                        help="Run directory name under hywp_runs_root "
                             "(e.g. bird_above_tree, or photorealistic/rigid/003).")
    parser.add_argument("--ckpt", required=True, type=int,
                        help="Checkpoint number (N for eval/ckpt<N>).")
    parser.add_argument("--instance", default=None,
                        help="Instance name (metadata only). Default: <run> with trailing _<suffix> stripped.")
    parser.add_argument("--poses", default=None,
                        help="Comma-separated subset of pose dir names "
                             "(e.g. 'seen_w_d,unseen_w_left'). Default: all present.")
    parser.add_argument("--pipeline", default="default",
                        help="ViPE pipeline config (default|dav3|no_vda|wide_angle|...). Default: default.")
    parser.add_argument("--include-gt", dest="include_gt", action="store_true", default=True,
                        help="Include the expected HYWP GT delta in the JSON. Default: on.")
    parser.add_argument("--no-include-gt", dest="include_gt", action="store_false",
                        help="Skip the expected GT delta.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose trajectory_eval.json already exists (resume).")
    parser.add_argument("--output-name", default="trajectory_eval.json",
                        help="Per-video JSON filename (default: trajectory_eval.json).")
    parser.add_argument("--poses-json", type=Path, default=None,
                        help="Poses JSON for pose dirs without pose_string.txt.")
    parser.add_argument("--flat-layout", action="store_true",
                        help="Pose dirs live directly under <hywp_runs_root>/<run>/ "
                             "(skip the eval/ckpt<n> nesting).")
    args = parser.parse_args()
    global POSES_JSON
    POSES_JSON = args.poses_json

    ckpt_dir = _resolve_ckpt_dir(args.hywp_runs_root, args.run, args.ckpt, flat=args.flat_layout)
    filter_poses = None
    if args.poses:
        filter_poses = [p.strip() for p in args.poses.split(",") if p.strip()]
    pose_dirs = _discover_pose_dirs(ckpt_dir, filter_poses)
    if not pose_dirs:
        raise SystemExit(f"No pose dirs found under {ckpt_dir} (filter={filter_poses})")

    instance = args.instance or _infer_instance_from_run(args.run)
    err_log = ckpt_dir / "trajectory_eval_errors.log"
    summary_rows: list[dict] = []
    skipped: list[str] = []

    logger.info(
        "Evaluating %d pose dir(s) under %s (instance=%s, pipeline=%s, include_gt=%s)",
        len(pose_dirs), ckpt_dir, instance, args.pipeline, args.include_gt,
    )

    for pose_dir in pose_dirs:
        video_path = pose_dir / "gen.mp4"
        out_json = pose_dir / args.output_name
        if not video_path.exists():
            logger.warning("skip %s: no gen.mp4", pose_dir.name)
            continue
        if args.skip_existing and out_json.exists():
            logger.info("skip %s: %s exists", pose_dir.name, out_json.name)
            try:
                with out_json.open() as f:
                    summary_rows.append(_summarize(json.load(f)))
            except Exception:                          # noqa: BLE001
                pass
            skipped.append(pose_dir.name)
            continue

        logger.info("▶ %s/%s", pose_dir.name, video_path.name)
        try:
            result = evaluate_video(
                video_path=video_path,
                pose_dir_name=pose_dir.name,
                pipeline=args.pipeline,
                include_gt=args.include_gt,
            )
            _write_json(out_json, result)
            summary_rows.append(_summarize(result))
            logger.info("  ✓ wrote %s", out_json)
        except Exception as e:                         # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("  ✗ failed on %s: %s", pose_dir.name, e)
            with err_log.open("a") as f:
                f.write(f"--- {pose_dir.name} / {video_path.name} ---\n{tb}\n")

    summary = {
        "run": args.run,
        "instance": instance,
        "ckpt": args.ckpt,
        "pipeline": args.pipeline,
        "include_gt": args.include_gt,
        "n_evaluated": len(summary_rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "results": summary_rows,
    }
    summary_path = ckpt_dir / "trajectory_eval_summary.json"
    _write_json(summary_path, summary)
    logger.info("Summary: %s (%d results)", summary_path, len(summary_rows))


if __name__ == "__main__":
    # Guard: this script must run inside the vipe conda env so that the `vipe`
    # package, torch, and scipy resolve. Emit a helpful message if not.
    if os.environ.get("CONDA_DEFAULT_ENV") not in (None, "", "vipe"):
        logger.warning(
            "Active conda env is '%s', not 'vipe'. This script expects the "
            "vipe env; imports may fail.",
            os.environ.get("CONDA_DEFAULT_ENV"),
        )
    main()
