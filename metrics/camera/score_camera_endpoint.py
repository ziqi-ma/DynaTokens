#!/usr/bin/env python3
"""Per-segment endpoint camera fidelity scoring.

Aggregates the trajectory_eval.json files written by vipe_pose_extraction.py
(found recursively under each --roots dir, one row per root) and scores each
camera segment at its endpoint:

  - endpoint_rot_err_deg : geodesic angle between segment-composed estimated
    rotation and segment-composed GT rotation (delta_estimated vs delta_expected
    already stored in trajectory_eval.json). Catches axis errors and systematic
    under/over-shoot that per-frame averaging dilutes.

  - endpoint_trans_err   : L2 distance between segment-composed estimated and
    GT translations.

For each segment we also report % of the chunk's full expected magnitude. For
"wrong-dimension" segments (rotation chunk where GT translation = 0, or vice
versa), the % uses an equivalent reference magnitude — so spurious motion in the silent dimension is
visible.

Usage
-----
    conda activate vipe
    python metrics/camera/score_camera_endpoint.py \\
        --roots outputs/runs/vbench outputs/eval/methods/hywp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Same GT motion constants and ceilings as the per-frame scorer.
FRAMES_PER_LATENT   = 4
GT_ROT_DEG_PER_LAT  = 3.0
GT_TRANS_PER_LAT    = 0.08



def _rotvec_deg_to_R(rv_deg: np.ndarray) -> np.ndarray:
    """Convert axis-angle in degrees → 3×3 rotation matrix (Rodrigues)."""
    rv = np.deg2rad(np.asarray(rv_deg, dtype=float))
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    axis = rv / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _chunk_n_latents(action: str, segment_len_frames: int) -> int:
    """Recover n (latents in this chunk) from the segment's frame span."""
    return max(1, segment_len_frames // FRAMES_PER_LATENT)


def score_video(eval_json_path: Path, trans_scale: float = 1.0,
                match_dim: bool = False) -> tuple[list, list, list, list]:
    """Return per-segment lists: rot_err_deg, rot_pct, trans_err, trans_pct."""
    with open(eval_json_path) as f:
        data = json.load(f)

    rot_raw, rot_pct, trans_raw, trans_pct = [], [], [], []

    for s in data["segments"]:
        action = s["action"]
        n_frames = int(s["end_frame"]) - int(s["start_frame"])
        n_lat = _chunk_n_latents(action, n_frames)
        chunk_rot_ref   = n_lat * GT_ROT_DEG_PER_LAT     # always available
        chunk_trans_ref = n_lat * GT_TRANS_PER_LAT

        e = s["delta_estimated"]
        g = s.get("delta_expected")
        if g is None:
            continue

        gt_trans_mag = float(np.linalg.norm(g["translation"]))
        is_rot_chunk = float(np.linalg.norm(g["rotation_rotvec_deg"])) > 1e-6
        is_trans_chunk = gt_trans_mag > 1e-6

        # Endpoint rotation error: geodesic between estimated and GT seg-end R.
        if (not match_dim) or is_rot_chunk:
            R_est = _rotvec_deg_to_R(e["rotation_rotvec_deg"])
            R_gt  = _rotvec_deg_to_R(g["rotation_rotvec_deg"])
            err_r = _geodesic_deg(R_est, R_gt)
            rot_raw.append(err_r)
            rot_pct.append(err_r / chunk_rot_ref * 100)

        # Endpoint translation error.
        if (not match_dim) or is_trans_chunk:
            t_est = np.asarray(e["translation"], dtype=float) / trans_scale
            t_gt  = np.asarray(g["translation"], dtype=float)
            err_t = float(np.linalg.norm(t_est - t_gt))
            trans_raw.append(err_t)
            trans_pct.append(err_t / chunk_trans_ref * 100)

    return rot_raw, rot_pct, trans_raw, trans_pct


# ── Aggregation ──────────────────────────────────────────────────────────────

def collect(
    roots: list[Path],
    exclude_pose_pairs: set[str] | None = None,
    trans_scales: dict[str, float] | None = None,
    match_dim: bool = False,
) -> dict:
    trans_scales = trans_scales or {}
    results: dict = {}

    def _ej_ok(ej: Path) -> bool:
        if exclude_pose_pairs and f"{ej.parents[1].name}/{ej.parent.name}" in exclude_pose_pairs:
            return False
        return True

    for root in roots:
        label = root.name
        bucket = results.setdefault(label, {"rr": [], "rp": [], "tr": [], "tp": []})
        for ej in sorted(root.rglob("trajectory_eval.json")):
            if not _ej_ok(ej):
                continue
            rr, rp, tr, tp = score_video(ej, trans_scales.get(label, 1.0), match_dim)
            bucket["rr"].extend(rr); bucket["rp"].extend(rp)
            bucket["tr"].extend(tr); bucket["tp"].extend(tp)

    return results


def combined_score(rr: list, tr: list, rot_max_deg: float, trans_max: float) -> tuple[float, float, float]:
    rot_err = float(np.mean(rr))
    trans_err = float(np.mean(tr))
    norm_r = 1.0 - np.clip(rot_err,   0.0, rot_max_deg) / rot_max_deg
    norm_t = 1.0 - np.clip(trans_err, 0.0, trans_max)   / trans_max
    return float(np.sqrt(norm_r * norm_t)), rot_err, trans_err


def print_results(results: dict, rot_max_deg: float, trans_max: float) -> None:
    hdr = (f"{'method':14s}  {'rot/seg(rad)':>12}  {'trans/seg':>9}  "
           f"{'score':>7}  {'n_seg':>6}")
    print(hdr); print("-" * len(hdr))
    for method, b in results.items():
        if not b["rr"]:
            continue
        score, rot_err, trans_err = combined_score(b["rr"], b["tr"], rot_max_deg, trans_max)
        print(
            f"{method:14s}  "
            f"{rot_err*np.pi/180:8.4f}      "
            f"{trans_err:9.4f}  "
            f"{score:7.4f}  "
            f"{len(b['rr']):6d}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--exclude-pose-pairs", nargs="+", default=None)
    parser.add_argument("--rot-max-deg", type=float, default=70.0,
                        help="Rotation error ceiling in degrees for combined score")
    parser.add_argument("--trans-max", type=float, default=4.5,
                        help="Translation error ceiling for combined score")
    parser.add_argument("--match-dim", action="store_true",
                        help="Score rotation only on rotation chunks and translation only on translation chunks.")
    parser.add_argument("--trans-scale", nargs="+", default=None,
                        help="Per-method translation divisor to correct unit difference across methods")
    args = parser.parse_args()

    trans_scales = {}
    if args.trans_scale:
        for kv in args.trans_scale:
            k, v = kv.split("=")
            trans_scales[k] = float(v)

    results = collect(
        roots=args.roots,
        exclude_pose_pairs=set(args.exclude_pose_pairs) if args.exclude_pose_pairs else None,
        trans_scales=trans_scales,
        match_dim=args.match_dim,
    )
    print_results(results, rot_max_deg=args.rot_max_deg, trans_max=args.trans_max)


if __name__ == "__main__":
    main()
