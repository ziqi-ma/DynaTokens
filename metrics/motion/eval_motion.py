"""WorldScore motion accuracy for rendered trajectories.

  motion_accuracy     did the prompted object move more than the camera-induced background?
                      (SEA-RAFT flow inside a GroundingDINO + SAM mask of the object)
Opt-in (--metrics all): motion_smoothness (VFIMamba, per pose segment), motion_magnitude.

Each EVAL_DIR is one scene: <eval_dir>/{seen,unseen}_<name>/gen.mp4 (+ pose_string.txt), as
written by dynatokens/inference.sh. The prompt comes from <eval_dir>/scene.json (or --prompt);
the object noun is extracted from it with spaCy (or --object-override).

Usage (worldscore env, see metrics/motion/README.md):
    python metrics/motion/eval_motion.py outputs/runs/<run>/eval/ckpt<step>

Writes <eval_dir>/<traj>/motion_eval.json and <eval_dir>/motion_eval_summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
WORLDSCORE_ROOT = REPO_ROOT / "third_party" / "WorldScore"

sys.path.insert(0, str(REPO_ROOT / "curation"))
from keyframe_utils import FRAMES_PER_LATENT, action_boundaries  # noqa: E402

# trajectory name -> pose string, filled from each video's pose_string.txt
_TRAJ_TO_POSE: dict[str, str] = {}


def _segments_for_traj(traj: str) -> list[tuple[str, int, int]] | None:
    """Return [(action, start_frame, end_frame), ...] or None if traj unknown."""
    pose = _TRAJ_TO_POSE.get(traj)
    if pose is None:
        return None
    bounds = action_boundaries(pose)   # [(action, end_frame), ...]
    out, prev = [], 0
    for action, end in bounds:
        out.append((action, prev, end))
        prev = end
    return out

# WorldScore metric checkpoints. These mirror the paths the metric classes
# hardcode internally (relative to WORLDSCORE_ROOT as cwd).
_CKPT_DIR = WORLDSCORE_ROOT / "worldscore/benchmark/metrics/checkpoints"
_GD_CONFIG = WORLDSCORE_ROOT / "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_GD_CKPT = _CKPT_DIR / "groundingdino_swint_ogc.pth"
_SAM_CKPT = _CKPT_DIR / "sam_vit_h_4b8939.pth"

_BOX_THRESH = 0.30
_TEXT_THRESH = 0.20

logger = logging.getLogger("eval_motion")


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCES_JSON + prompt/object resolution
# ─────────────────────────────────────────────────────────────────────────────
_NLP = None


def _spacy_nlp():
    global _NLP
    if _NLP is None:
        import spacy  # noqa: WPS433
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_object(prompt: str) -> str | None:
    """Heuristic head-noun extraction. Returns lowercase noun or None."""
    if not prompt:
        return None
    nlp = _spacy_nlp()
    doc = nlp(prompt)
    # Prefer the subject-like noun in the first sentence; fall back to first noun.
    for sent in doc.sents:
        for tok in sent:
            if tok.dep_ in {"nsubj", "nsubjpass"} and tok.pos_ in {"NOUN", "PROPN"}:
                return tok.lemma_.lower()
        for tok in sent:
            if tok.pos_ in {"NOUN", "PROPN"}:
                return tok.lemma_.lower()
    return None


@contextmanager
def decoded_frames(video_path: Path):
    """Yield a list of frame paths, decoded into a tmp dir (auto-cleaned)."""
    tmp = Path(tempfile.mkdtemp(prefix="motion_eval_"))
    try:
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"{n:03d}.png"), frame)
            n += 1
        cap.release()
        if n == 0:
            raise RuntimeError(f"no frames decoded from {video_path}")
        frame_paths = sorted(str(p) for p in frames_dir.glob("*.png"))
        yield tmp, frame_paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Mask generation (GroundingDINO box + SAM-ViT-H mask) for first frame
# ─────────────────────────────────────────────────────────────────────────────
class MaskGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info("loading GroundingDINO...")
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        cfg = SLConfig.fromfile(str(_GD_CONFIG))
        cfg.device = device
        cfg.bert_base_uncased_path = None
        gd = build_model(cfg)
        ckpt = torch.load(str(_GD_CKPT), map_location="cpu")
        gd.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self._gd = gd.eval().to(device)

        logger.info("loading SAM ViT-H...")
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=str(_SAM_CKPT))
        self._sam = SamPredictor(sam.to(device))

        # GroundingDINO image transform.
        import groundingdino.datasets.transforms as T  # noqa: WPS433
        self._gd_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _detect_box(self, image_pil: Image.Image, caption: str) -> np.ndarray | None:
        cap = caption.strip().lower()
        if not cap.endswith("."):
            cap += "."
        image_t, _ = self._gd_transform(image_pil, None)
        image_t = image_t.to(self.device)
        with torch.no_grad():
            out = self._gd(image_t[None], captions=[cap])
        logits = out["pred_logits"].cpu().sigmoid()[0]
        boxes = out["pred_boxes"].cpu()[0]
        score_per_box = logits.max(dim=1)[0]
        keep = score_per_box > _BOX_THRESH
        if not keep.any():
            return None
        best = int(torch.argmax(score_per_box[keep]))
        cx, cy, bw, bh = boxes[keep][best].tolist()
        W, H = image_pil.size
        return np.array([
            (cx - bw / 2) * W,
            (cy - bh / 2) * H,
            (cx + bw / 2) * W,
            (cy + bh / 2) * H,
        ])

    def first_frame_mask(self, frame_path: Path, obj: str, out_dir: Path) -> Path | None:
        image_pil = Image.open(frame_path).convert("RGB")
        box = self._detect_box(image_pil, obj)
        if box is None:
            logger.warning("GroundingDINO found no box for '%s' in %s", obj, frame_path)
            return None
        image_bgr = cv2.imread(str(frame_path))
        self._sam.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        masks, _, _ = self._sam.predict(
            point_coords=None, point_labels=None,
            box=box[None, :], multimask_output=False,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / "000.png"
        cv2.imwrite(str(mask_path), (masks[0].astype(np.uint8)) * 255)
        return mask_path


# ─────────────────────────────────────────────────────────────────────────────
# Metric runners (lazy-load on first use)
# ─────────────────────────────────────────────────────────────────────────────
class MetricBundle:
    """Holds the WorldScore metric instances; each loads on first use."""

    def __init__(self):
        self._motion_accuracy = None
        self._motion_smoothness = None

    @property
    def motion_accuracy(self):
        if self._motion_accuracy is None:
            logger.info("loading MotionAccuracyMetric (SEA-RAFT + SAM2)...")
            from worldscore.benchmark.metrics.third_party.motion_accuracy_metrics import (
                MotionAccuracyMetric,
            )
            # generate_type is only used for the GroundingDINO branch which we
            # do NOT trigger here (we provide masks ourselves). Pass "i2v" so
            # that branch is skipped entirely.
            self._motion_accuracy = MotionAccuracyMetric(generate_type="i2v")
        return self._motion_accuracy

    @property
    def motion_smoothness(self):
        if self._motion_smoothness is None:
            logger.info("loading MotionSmoothnessMetric (VFIMamba)...")
            from worldscore.benchmark.metrics.third_party.motion_smoothness_metrics import (
                MotionSmoothnessMetric,
            )
            self._motion_smoothness = MotionSmoothnessMetric()
        return self._motion_smoothness


# ─────────────────────────────────────────────────────────────────────────────
# Per-video evaluation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VideoResult:
    video_path: Path
    instance: str
    bucket: str            # "seen" / "unseen" / "train" / "test"
    trajectory: str        # the WASD pose dir name (e.g. "w_d", "a2")
    prompt: str
    obj: str | None
    motion_accuracy: dict | None = None              # {"score": ..., "variants": {...}, ...}
    motion_magnitude: dict | None = None              # {"score": float, "n_pairs": int}
    motion_smoothness: dict | None = None            # {"score": [mse, ssim, lpips], "per_segment": [...], ...}
    error: str | None = None
    extras: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "video": self.video_path.name,
            "video_path": str(self.video_path),
            "instance": self.instance,
            "bucket": self.bucket,
            "trajectory": self.trajectory,
            "prompt": self.prompt,
            "object": self.obj,
            "metrics": {
                "motion_accuracy": self.motion_accuracy,
                "motion_magnitude": self.motion_magnitude,
                "motion_smoothness": self.motion_smoothness,
            },
            "error": self.error,
            **self.extras,
        }


_ALL_METRICS = (
    "motion_accuracy",
    "motion_magnitude",
    "motion_smoothness",
)
_DEFAULT_METRICS = ("motion_accuracy",)

_DEFAULT_SCORE_VARIANT = "obj_max_minus_bg_median"
_VARIANT_FORMULAS = {
    # Each formula takes the four per-pair stat lists and returns the per-pair
    # series; the headline ``score`` is the mean of that series.
    "max_diff":                lambda om, oe, oa, bm, be, ba: [a - b for a, b in zip(oa, ba)],
    "mean_diff":               lambda om, oe, oa, bm, be, ba: [a - b for a, b in zip(om, bm)],
    "median_diff":             lambda om, oe, oa, bm, be, ba: [a - b for a, b in zip(oe, be)],
    "obj_max":                 lambda om, oe, oa, bm, be, ba: list(oa),
    "obj_mean":                lambda om, oe, oa, bm, be, ba: list(om),
    "obj_max_minus_bg_median": lambda om, oe, oa, bm, be, ba: [a - b for a, b in zip(oa, be)],
    "obj_mean_minus_bg_median":lambda om, oe, oa, bm, be, ba: [a - b for a, b in zip(om, be)],
}


def _compute_motion_accuracy_variants(
    metric,                 # MotionAccuracyMetric instance (gives us SAM2 + SEA-RAFT)
    frame_paths: list[str],
    mask_path: Path,
    obj: str,
    score_variant: str = _DEFAULT_SCORE_VARIANT,
) -> dict:
    """Run the same SAM2 propagation + SEA-RAFT flow as
    ``MotionAccuracyMetric._compute_scores`` but record several reductions
    of the per-frame (object, background) flow distribution.

    The headline ``score`` is selected via ``score_variant``; all variants are
    also written for inspection. Default = ``obj_max_minus_bg_median``, which
    is more robust than WorldScore's ``max(obj) − max(bg)`` in our setting:
    with a 96% background and a moving camera, ``max(bg)`` is dominated by a
    single high-flow edge pixel that scales with camera-following sharpness,
    not with scene dynamics. ``bg_median`` is a much more stable estimate of
    the camera-induced floor, so subtracting it gives "peak object motion
    above what the camera alone would produce".

    Pass ``score_variant="max_diff"`` for the WorldScore-published metric.

    Returns a dict suitable for embedding under
    ``motion_eval.json::metrics.motion_accuracy``.
    """
    from worldscore.benchmark.metrics.third_party.motion_accuracy_metrics import (
        vos_inference, mask_resize,
    )
    video_dir = os.path.dirname(frame_paths[0])
    mask_dir = os.path.dirname(str(mask_path))
    masks = vos_inference(
        predictor=metric._predictor,
        video_dir=video_dir,
        input_mask=mask_dir,
        score_thresh=0.0,
        per_obj_png_file=True,
    )

    # Per-pair (obj, bg) flow distribution stats.
    per_pair = []
    obj_means, obj_meds, obj_maxs = [], [], []
    bg_means, bg_meds, bg_maxs = [], [], []
    full_meds = []                # full-frame median magnitude → motion_magnitude
    mask_areas = []
    skipped_pairs = 0

    with torch.no_grad():
        for i, (f1, f2, mask) in enumerate(zip(frame_paths[:-1], frame_paths[1:], masks[:-1])):
            img1 = metric.load_image(f1)
            img2 = metric.load_image(f2)
            if mask.shape != img1.shape[2:]:
                mask = mask_resize(mask, img1.shape[2:])
            flow = metric._compute_flow(img1, img2)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            full_meds.append(float(np.median(mag)))   # the WorldScore motion_magnitude
            obj = mag[mask]
            bg = mag[~mask]
            if obj.size == 0 or bg.size == 0:
                skipped_pairs += 1
                continue
            mask_areas.append(float(mask.sum()) / mask.size)
            obj_means.append(float(obj.mean()));  bg_means.append(float(bg.mean()))
            obj_meds.append(float(np.median(obj))); bg_meds.append(float(np.median(bg)))
            obj_maxs.append(float(obj.max()));    bg_maxs.append(float(bg.max()))
            per_pair.append({
                "frame_a": i, "frame_b": i + 1,
                "mask_area_frac": mask_areas[-1],
                "full_median": full_meds[-1],
                "obj": {"mean": obj_means[-1], "median": obj_meds[-1], "max": obj_maxs[-1]},
                "bg":  {"mean": bg_means[-1],  "median": bg_meds[-1],  "max": bg_maxs[-1]},
            })

    if not obj_means:
        return {"error": "all pairs skipped (empty obj/bg mask)",
                "skipped_pairs": skipped_pairs}

    def _agg(xs): return float(np.mean(xs))

    if score_variant not in _VARIANT_FORMULAS:
        raise ValueError(f"unknown score_variant '{score_variant}'; "
                         f"expected one of {sorted(_VARIANT_FORMULAS)}")
    variants = {
        name: _agg(formula(obj_means, obj_meds, obj_maxs, bg_means, bg_meds, bg_maxs))
        for name, formula in _VARIANT_FORMULAS.items()
    }

    summary = {
        "score": variants[score_variant],
        "score_variant": score_variant,
        "variants": variants,
        "obj_mean":     _agg(obj_means),
        "obj_median":   _agg(obj_meds),
        "obj_max":      _agg(obj_maxs),
        "bg_mean":      _agg(bg_means),
        "bg_median":    _agg(bg_meds),
        "bg_max":       _agg(bg_maxs),
        "mask_area_frac_mean": _agg(mask_areas),
        "mask_area_frac_min":  float(np.min(mask_areas)),
        "mask_area_frac_max":  float(np.max(mask_areas)),
        "n_pairs": len(obj_means),
        "n_skipped_pairs": skipped_pairs,
        "per_pair": per_pair,
        "notes": ("Default score_variant = obj_max_minus_bg_median (robust in "
                  "camera-controlled videos). Pass --motion-accuracy-score "
                  "max_diff for the WorldScore-published metric "
                  "(max(obj) - max(bg)). The per_pair list captures the "
                  "frame-by-frame stats so you can plot a time-series "
                  "(useful when one bad pair dominates the mean)."),
    }
    # Side-channel: the SEA-RAFT loop also gives us full-frame median per pair,
    # which is exactly WorldScore's motion_magnitude (utils.py:176, flow_metrics.py:87).
    # Returned separately so evaluate_video can write it under its own metric key.
    summary["_full_median_per_pair"] = full_meds
    return summary


def _compute_motion_smoothness_per_segment(
    motion_smoothness_metric,
    frame_paths: list[str],
    trajectory: str,
) -> dict:
    """Run VFIMamba once per pose-string segment and return per-segment +
    headline (mean of per-segment) scores.

    Falls back to a single full-video pass when the trajectory name isn't in
    keyframe_utils or all segments are below VFIMamba's 3-frame minimum.
    """
    def _full_video() -> dict:
        s = list(motion_smoothness_metric._compute_scores(frame_paths))
        return {
            "score": [float(x) for x in s],
            "components": ["mse", "ssim", "lpips"],
            "mode": "full_video",
            "pose_string": _TRAJ_TO_POSE.get(trajectory),
            "per_segment": None,
            "n_segments": 0,
        }

    segments = _segments_for_traj(trajectory)
    if segments is None:
        logger.warning("no pose string for traj '%s'; full-video smoothness only",
                       trajectory)
        return _full_video()

    pose_string = _TRAJ_TO_POSE[trajectory]
    per_seg = []
    n = len(frame_paths)
    for action, start, end in segments:
        # Inclusive endpoints; pose_string boundaries are 4 px/latent on a
        # 1-anchor + 60-generated frame layout. Clamp to actual frame count
        # in case the video is shorter than the nominal 61.
        s = max(0, min(start, n - 1))
        e = max(0, min(end, n - 1))
        seg_frames = frame_paths[s : e + 1]
        if len(seg_frames) < 3:
            per_seg.append({
                "action": action, "start_frame": s, "end_frame": e,
                "n_frames": len(seg_frames), "score": None,
                "skipped": "too few frames for VFIMamba (<3)",
            })
            continue
        try:
            score = list(motion_smoothness_metric._compute_scores(seg_frames))
            per_seg.append({
                "action": action, "start_frame": s, "end_frame": e,
                "n_frames": len(seg_frames),
                "score": [float(x) for x in score],
            })
        except Exception as e_seg:                      # noqa: BLE001
            logger.warning("seg %s [%d,%d] failed: %s", action, s, e, e_seg)
            per_seg.append({
                "action": action, "start_frame": s, "end_frame": e,
                "n_frames": len(seg_frames), "score": None,
                "error": str(e_seg),
            })

    valid = [seg["score"] for seg in per_seg if seg["score"] is not None]
    if not valid:
        logger.warning("no valid segments for traj '%s'; falling back to full-video",
                       trajectory)
        return _full_video()

    mean_score = np.mean(np.array(valid), axis=0).tolist()
    return {
        "score": [float(x) for x in mean_score],
        "components": ["mse", "ssim", "lpips"],
        "mode": "per_segment_mean",
        "pose_string": pose_string,
        "per_segment": per_seg,
        "n_segments": len(valid),
        "notes": ("Headline score is the mean across pose-string segments. "
                  "Per-segment passes drop the cross-action VFIMamba triplet, "
                  "so within-action smoothness is isolated from transition "
                  "jumps. See per_segment for the per-action breakdown."),
    }


def evaluate_video(
    video_path: Path,
    *,
    instance: str,
    bucket: str,
    trajectory: str,
    prompt: str,
    obj: str | None,
    metrics: MetricBundle,
    masker: MaskGenerator | None,
    enabled_metrics: frozenset[str] = frozenset(_DEFAULT_METRICS),
    motion_accuracy_score: str = _DEFAULT_SCORE_VARIANT,
) -> VideoResult:
    res = VideoResult(
        video_path=video_path,
        instance=instance,
        bucket=bucket,
        trajectory=trajectory,
        prompt=prompt,
        obj=obj,
    )
    with decoded_frames(video_path) as (tmp_root, frame_paths):
        # ── motion_smoothness ──────────────────────────────────────────────
        if "motion_smoothness" in enabled_metrics:
            try:
                res.motion_smoothness = _compute_motion_smoothness_per_segment(
                    metrics.motion_smoothness, frame_paths, trajectory,
                )
            except Exception as e:                      # noqa: BLE001
                logger.error("motion_smoothness failed on %s: %s", video_path.name, e)
                res.error = (res.error or "") + f"motion_smoothness: {e}; "

        # ── motion_accuracy + motion_magnitude (shared SEA-RAFT loop) ──────
        wants_ma = "motion_accuracy"  in enabled_metrics
        wants_mm = "motion_magnitude" in enabled_metrics
        if (wants_ma or wants_mm) and obj and masker is not None:
            try:
                mask_path = masker.first_frame_mask(
                    Path(frame_paths[0]), obj, tmp_root / "masks",
                )
                if mask_path is None:
                    res.extras["motion_accuracy_skipped"] = (
                        f"GroundingDINO found no '{obj}' in first frame"
                    )
                    return res
                ma = _compute_motion_accuracy_variants(
                    metrics.motion_accuracy, frame_paths, mask_path, obj,
                    score_variant=motion_accuracy_score,
                )
                # Split off motion_magnitude (computed for free in the same flow loop).
                full_meds = ma.pop("_full_median_per_pair", None)
                if wants_mm and full_meds:
                    res.motion_magnitude = {
                        "score": float(np.mean(full_meds)),
                        "per_pair_median": full_meds,
                        "n_pairs": len(full_meds),
                        "metric": "optical_flow",
                        "notes": ("WorldScore motion_magnitude: mean over "
                                  "frame-pairs of (median over all pixels of "
                                  "SEA-RAFT flow magnitude). Confounded by "
                                  "camera motion in our setting."),
                    }
                if wants_ma:
                    res.motion_accuracy = ma
            except Exception as e:                      # noqa: BLE001
                logger.error("motion_accuracy/magnitude failed on %s: %s",
                             video_path.name, e)
                res.error = (res.error or "") + f"motion_accuracy: {e}; "

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Discovery: find all (video_path, instance, bucket, trajectory) tuples
# ─────────────────────────────────────────────────────────────────────────────
def _summarize(rows: list[dict]) -> dict:
    def _agg(metric_key: str) -> dict | None:
        vals = [r["metrics"][metric_key]["score"] for r in rows
                if r["metrics"].get(metric_key) is not None]
        if not vals:
            return None
        if isinstance(vals[0], list):
            arr = np.array(vals)
            return {"mean": arr.mean(axis=0).tolist(),
                    "n": int(arr.shape[0])}
        return {"mean": float(np.mean(vals)), "n": len(vals)}
    return {
        "n_videos": len(rows),
        "motion_accuracy": _agg("motion_accuracy"),
        "motion_magnitude": _agg("motion_magnitude"),
        "motion_smoothness": _agg("motion_smoothness"),
        "by_bucket": {
            bucket: [r for r in rows if r["bucket"] == bucket]
            for bucket in sorted({r["bucket"] for r in rows})
        },
    }


def main() -> None:
    global WORLDSCORE_ROOT, _CKPT_DIR, _GD_CONFIG, _GD_CKPT, _SAM_CKPT
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("eval_dirs", nargs="+", type=Path, help="Eval dirs, one per scene")
    p.add_argument("--prompt", default=None, help="Scene prompt (default: <eval_dir>/scene.json)")
    p.add_argument("--worldscore-root", type=Path, default=WORLDSCORE_ROOT,
                   help=f"WorldScore checkout (default: {WORLDSCORE_ROOT})")
    p.add_argument("--metrics",
                   default=",".join(_DEFAULT_METRICS),
                   help="Comma-separated subset of {" + ",".join(_ALL_METRICS) + "}, or 'all'. "
                        f"Default: {','.join(_DEFAULT_METRICS)}.")
    p.add_argument("--object-override", default=None,
                   help="Object noun for the motion_accuracy mask (default: spaCy from the prompt).")
    p.add_argument("--motion-accuracy-score",
                   default=_DEFAULT_SCORE_VARIANT,
                   choices=sorted(_VARIANT_FORMULAS),
                   help=f"Headline motion_accuracy variant (default: {_DEFAULT_SCORE_VARIANT}; "
                        "'max_diff' is WorldScore's published formula). All variants are recorded.")
    args = p.parse_args()

    if args.metrics.strip().lower() == "all":
        enabled = frozenset(_ALL_METRICS)
    else:
        requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = [m for m in requested if m not in _ALL_METRICS]
        if unknown:
            p.error(f"unknown metric(s): {unknown}; valid: {list(_ALL_METRICS)}")
        enabled = frozenset(requested)

    eval_dirs = [d.resolve() for d in args.eval_dirs]
    # WorldScore metrics use cwd-relative checkpoint paths. Rebase the
    # module-level checkpoint/config paths too, so --worldscore-root works.
    WORLDSCORE_ROOT = args.worldscore_root.resolve()
    _CKPT_DIR = WORLDSCORE_ROOT / "worldscore/benchmark/metrics/checkpoints"
    _GD_CONFIG = WORLDSCORE_ROOT / "worldscore/benchmark/metrics/third_party/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    _GD_CKPT = _CKPT_DIR / "groundingdino_swint_ogc.pth"
    _SAM_CKPT = _CKPT_DIR / "sam_vit_h_4b8939.pth"
    os.chdir(WORLDSCORE_ROOT)
    sys.path.insert(0, str(WORLDSCORE_ROOT))

    metrics = MetricBundle()
    needs_mask = ("motion_accuracy" in enabled) or ("motion_magnitude" in enabled)
    masker = (MaskGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
              if needs_mask else None)

    for eval_dir in eval_dirs:
        scene_json = eval_dir / "scene.json"
        if args.prompt is not None:
            prompt = args.prompt
        elif scene_json.is_file():
            prompt = json.loads(scene_json.read_text())["prompt"]
        elif args.object_override:
            prompt = ""  # only needed for spaCy object extraction, overridden anyway
        else:
            sys.exit(f"{eval_dir}: no scene.json found; pass --prompt or --object-override")
        obj = args.object_override or (extract_object(prompt) if needs_mask else None)
        logger.info("%s  prompt=%r  obj=%s", eval_dir, prompt, obj)
        rows = []
        videos = sorted(eval_dir.glob("*/gen.mp4"))
        if not videos and (eval_dir / "gen.mp4").is_file():
            videos = [eval_dir / "gen.mp4"]  # a single pose dir was passed
        if not videos:
            sys.exit(f"{eval_dir}: no gen.mp4 found (expected <eval_dir>/<pose>/gen.mp4)")
        for video_path in videos:
            bucket, _, trajectory = video_path.parent.name.partition("_")
            pose_file = video_path.parent / "pose_string.txt"
            if pose_file.is_file():
                _TRAJ_TO_POSE[trajectory] = pose_file.read_text().strip()
            res = evaluate_video(
                video_path,
                instance=eval_dir.name, bucket=bucket, trajectory=trajectory,
                prompt=prompt, obj=obj,
                metrics=metrics, masker=masker,
                enabled_metrics=enabled,
                motion_accuracy_score=args.motion_accuracy_score,
            )
            (video_path.parent / "motion_eval.json").write_text(json.dumps(res.to_json(), indent=2))
            rows.append(res.to_json())
        (eval_dir / "motion_eval_summary.json").write_text(json.dumps(_summarize(rows), indent=2))
        logger.info("wrote %s", eval_dir / "motion_eval_summary.json")


if __name__ == "__main__":
    main()
