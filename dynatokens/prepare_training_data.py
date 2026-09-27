"""
Encode the curated supervision videos of one scene into DynaToken training data.

Input: a curation scene dir (see curation/README.md)
    <scene-dir>/init_16x9.png
    <scene-dir>/{train,test}/<name>/stitched.mp4   (+ stitched_review.json)
and the poses JSON used for curation:
    {"train": {"<name>": "<pose string>", ...}, "test": {"<name>": "<pose string>", ...}}

Output (<output-dir>, consumed by train.sh / inference.sh):
    train/<name>/{latents_textprompt.pt, pose.json, pose_string.txt}   training trajectories
    test/<name>/{pose.json, pose_string.txt}                           held-out trajectories (eval only)
    shared/hunyuan_neg_{prompt,byt5_prompt}.pt                         negative-prompt embeddings
    train_all_textprompt.json                                          training manifest
    scene.json                                                         prompt, image, window length

Train trajectories are kept if their video exists and passed the curation review
(stitched_review.json, if present; disable with --no-review-filter), or, with
--passing-list, if listed there (one video path per line).

Usage (from the repo root):
  # one scene
  PYTHONPATH=model/hywp python dynatokens/prepare_training_data.py \\
      --scene-dir  outputs/curation/my_scenes/bird \\
      --poses      my_poses.json \\
      --prompt     "A bird flies from the branch to the ground." \\
      --output-dir outputs/training_data/bird

  # every scene of a curation run (same scenes JSON as curation/run.sh)
  PYTHONPATH=model/hywp python dynatokens/prepare_training_data.py \\
      --scenes my_scenes.json --curation-root outputs/curation/my_scenes \\
      --poses  my_poses.json  --output-dir outputs/training_data/my_scenes
"""

import argparse
import json
import os

import numpy as np
import torch
from torchvision import transforms

HEIGHT, WIDTH = 480, 832
LATENT_STRIDE = 4  # video frames per latent


# ──────────────────────────────────────────────────────────────────────────────
# Poses
# ──────────────────────────────────────────────────────────────────────────────

def load_poses(poses_json: str) -> dict[str, dict[str, str]]:
    """{"train": {name: pose_string}, "test": {name: pose_string}}"""
    with open(poses_json) as f:
        poses = json.load(f)
    unknown = set(poses) - {"train", "test"}
    assert not unknown, f"{poses_json}: unexpected keys {sorted(unknown)} (use 'train' / 'test')"
    return {"train": poses.get("train", {}), "test": poses.get("test", {})}


def pose_num_latents(pose_string: str) -> int:
    """Latent frames of a trajectory: one per motion step plus the initial frame."""
    return 1 + sum(int(seg.strip().rsplit("-", 1)[1]) for seg in pose_string.split(",") if seg.strip())


def pose_string_to_pose_json(pose_string: str) -> dict:
    from hyvideo.generate import parse_pose_string
    from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local

    raw_intrinsic = np.array([
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ])
    norm_intrinsic = raw_intrinsic.copy()
    norm_intrinsic[0, 0] /= raw_intrinsic[0, 2] * 2
    norm_intrinsic[1, 1] /= raw_intrinsic[1, 2] * 2
    norm_intrinsic[0, 2] = 0.5
    norm_intrinsic[1, 2] = 0.5

    motions = parse_pose_string(pose_string)
    c2w_list = generate_camera_trajectory_local(motions)

    pose_dict = {}
    for i, c2w in enumerate(c2w_list):
        w2c = np.linalg.inv(c2w)
        pose_dict[str(i)] = {"w2c": w2c.tolist(), "intrinsic": norm_intrinsic.tolist()}
    return pose_dict


def write_pose(out_dir: str, pose_string: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pose_path = os.path.join(out_dir, "pose.json")
    with open(pose_path, "w") as f:
        json.dump(pose_string_to_pose_json(pose_string), f, indent=2)
    with open(os.path.join(out_dir, "pose_string.txt"), "w") as f:
        f.write(pose_string + "\n")
    return pose_path


def select_train_trajectories(scene_dir: str, train_poses: dict, video_name: str,
                              passing_list: str | None, review_filter: bool) -> list[tuple[str, str, str]]:
    """Return [(name, pose_string, video_path)] of usable training trajectories."""
    passing = None
    if passing_list:
        with open(passing_list) as f:
            passing = {os.path.basename(os.path.dirname(l.strip())) for l in f if l.strip()}
    selected = []
    for name, pose_string in train_poses.items():
        video_path = os.path.join(scene_dir, "train", name, video_name)
        if not os.path.isfile(video_path):
            print(f"  [SKIP] train/{name}: {video_name} not found")
            continue
        if passing is not None:
            if name not in passing:
                print(f"  [SKIP] train/{name}: not in {passing_list}")
                continue
        elif review_filter:
            review = os.path.join(os.path.dirname(video_path), "stitched_review.json")
            if os.path.isfile(review):
                with open(review) as f:
                    if not json.load(f).get("passed", False):
                        print(f"  [SKIP] train/{name}: failed curation review")
                        continue
        selected.append((name, pose_string, video_path))
    return selected


def load_video_frames(video_path: str, target_frames: int) -> torch.Tensor:
    """Load video as float32 tensor [C, T, H, W] in [-1, 1]."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    while len(frames) < target_frames:
        frames.append(frames[-1].copy())
    frames = frames[:target_frames]

    arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2)  # [C, T, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SceneEncoder:
    """HunyuanVideo-1.5 VAE, text, vision and byT5 encoders, loaded once."""

    def __init__(self, hunyuan_checkpoint: str, device: str = "cuda:0"):
        ckpt = hunyuan_checkpoint
        self.device = device = torch.device(device)

        print("Loading VAE...")
        from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache
        self.vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
            os.path.join(ckpt, "vae"), torch_dtype=torch.bfloat16).to(device)
        self.vae.eval()
        self.scaling_factor = self.vae.config.scaling_factor

        print("Loading text encoder...")
        from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=os.path.join(ckpt, "text_encoder", "llm"),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            device=device,
        )

        print("Loading vision encoder...")
        from hyvideo.models.vision_encoder import VisionEncoder
        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=os.path.join(ckpt, "vision_encoder", "siglip"),
            device=device,
        )

        print("Loading byT5...")
        from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
        glyph_root = os.path.join(ckpt, "text_encoder", "Glyph-SDXL-v2")
        byt5_kwargs = load_glyph_byT5_v2({
            "byT5_google_path": os.path.join(ckpt, "text_encoder", "byt5-small"),
            "byT5_ckpt_path": os.path.join(glyph_root, "checkpoints", "byt5_model.pt"),
            "multilingual_prompt_format_color_path": os.path.join(glyph_root, "assets", "color_idx.json"),
            "multilingual_prompt_format_font_path": os.path.join(glyph_root, "assets", "multilingual_10-lang_idx.json"),
            "byt5_max_length": 256,
        }, device=device)
        self.byt5_max_length = byt5_kwargs["byt5_max_length"]

    @torch.no_grad()
    def encode_text(self, prompt: str, uncond: bool = False):
        inputs = self.text_encoder.text2tokens(prompt, data_type="video", max_length=1000)
        out = self.text_encoder.encode(inputs, data_type="video", device=self.device, is_uncond=uncond)
        return out.hidden_state.cpu(), out.attention_mask.cpu()

    def empty_byt5(self):
        return (torch.zeros(1, self.byt5_max_length, 1472),
                torch.zeros(1, self.byt5_max_length, dtype=torch.int64))

    @torch.no_grad()
    def encode_image(self, image_path: str):
        """Return (vision_states, image_cond latent) for the reference image."""
        from PIL import Image
        from hyvideo.utils.data_utils import resize_and_center_crop
        ref_image = Image.open(image_path).convert("RGB")
        input_image_np = resize_and_center_crop(np.array(ref_image), target_width=WIDTH, target_height=HEIGHT)
        vision_states = self.vision_encoder.encode_images(input_image_np).last_hidden_state.to(
            dtype=torch.bfloat16).cpu()
        img_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])(Image.fromarray(input_image_np))
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_cond = self.vae.encode(img_tensor).latent_dist.mode() * self.scaling_factor
        return vision_states, image_cond.cpu()

    @torch.no_grad()
    def encode_video(self, video_path: str, num_frames: int):
        video = load_video_frames(video_path, num_frames).unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent = self.vae.encode(video).latent_dist.mode() * self.scaling_factor
        return latent.cpu()


def prepare_scene(encoder: SceneEncoder, scene_dir: str, poses: dict, prompt: str, output_dir: str,
                  image_path: str | None = None, video_name: str = "stitched.mp4",
                  passing_list: str | None = None, review_filter: bool = True) -> int:
    """Encode one scene into output_dir. Returns the number of training trajectories."""
    image_path = image_path or os.path.join(scene_dir, "init_16x9.png")
    all_poses = list(poses["train"].values()) + list(poses["test"].values())
    lengths = {pose_num_latents(p) for p in all_poses}
    assert len(lengths) == 1, f"All poses must have the same number of latents, got {sorted(lengths)}"
    num_latents = lengths.pop()
    assert num_latents % 4 == 0, f"Poses span {num_latents} latents (motion steps + 1); must be a multiple of 4"
    num_frames = (num_latents - 1) * LATENT_STRIDE + 1
    os.makedirs(output_dir, exist_ok=True)

    samples = select_train_trajectories(scene_dir, poses["train"], video_name, passing_list, review_filter)
    print(f"{scene_dir}: {len(samples)}/{len(poses['train'])} training trajectories, "
          f"{num_latents} latents ({num_frames} frames)")

    # held-out trajectories: camera poses only (rendered from the first frame at eval time)
    for name, pose_string in poses["test"].items():
        write_pose(os.path.join(output_dir, "test", name), pose_string)

    # shared negative-prompt embeddings
    shared_dir = os.path.join(output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    neg_embeds, neg_mask = encoder.encode_text("", uncond=True)
    neg_byt5_states, neg_byt5_mask = encoder.empty_byt5()
    torch.save({"negative_prompt_embeds": neg_embeds, "negative_prompt_mask": neg_mask},
               os.path.join(shared_dir, "hunyuan_neg_prompt.pt"))
    torch.save({"byt5_text_states": neg_byt5_states, "byt5_text_mask": neg_byt5_mask},
               os.path.join(shared_dir, "hunyuan_neg_byt5_prompt.pt"))

    prompt_embeds = prompt_mask = vision_states = image_cond = None
    entries = []
    for name, pose_string, video_path in samples:
        out_dir = os.path.join(output_dir, "train", name)
        latent_out = os.path.join(out_dir, "latents_textprompt.pt")
        pose_out = os.path.join(out_dir, "pose.json")
        entries.append({"latent_path": os.path.abspath(latent_out), "pose_path": os.path.abspath(pose_out)})
        if os.path.exists(latent_out) and os.path.exists(pose_out):
            print(f"  [CACHED] train/{name}")
            continue
        if prompt_embeds is None:  # encode scene conditioning once, lazily
            print(f"  Encoding prompt: {prompt!r}")
            prompt_embeds, prompt_mask = encoder.encode_text(prompt)
            print(f"  Encoding reference image: {image_path}")
            vision_states, image_cond = encoder.encode_image(image_path)
        write_pose(out_dir, pose_string)
        latent = encoder.encode_video(video_path, num_frames)
        byt5_states, byt5_mask = encoder.empty_byt5()
        torch.save({
            "latent":           latent,
            "prompt_embeds":    prompt_embeds,
            "prompt_mask":      prompt_mask,
            "image_cond":       image_cond,
            "vision_states":    vision_states,
            "byt5_text_states": byt5_states,
            "byt5_text_mask":   byt5_mask,
        }, latent_out)
        print(f"  train/{name}: latent {tuple(latent.shape)}")

    with open(os.path.join(output_dir, "train_all_textprompt.json"), "w") as f:
        json.dump(entries, f, indent=2)
    with open(os.path.join(output_dir, "scene.json"), "w") as f:
        json.dump({"prompt": prompt,
                   "image_path": os.path.abspath(image_path),
                   "scene_dir": os.path.abspath(scene_dir),
                   "video_name": video_name,
                   "window_frames": num_latents}, f, indent=2)
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", help="Curation output dir of one scene (init_16x9.png, train/, test/)")
    parser.add_argument("--prompt", help="Text prompt of that scene (action description)")
    parser.add_argument("--scenes", help="Scenes JSON [{image, prompt, output}] to prepare every scene")
    parser.add_argument("--curation-root", help="With --scenes: curation OUTPUT_ROOT (scene dirs are <root>/<output>)")
    parser.add_argument("--poses", required=True,
                        help='Poses JSON: {"train": {name: pose_string}, "test": {name: pose_string}}')
    parser.add_argument("--output-dir", required=True,
                        help="Training data dir (with --scenes: root, one subdir per scene)")
    parser.add_argument("--image-path", help="Reference image (default: <scene-dir>/init_16x9.png)")
    parser.add_argument("--video-name", default="stitched.mp4",
                        help="Video filename in each trajectory dir (default: stitched.mp4)")
    parser.add_argument("--passing-list", help="Use only the videos listed in this file (one path per line)")
    parser.add_argument("--no-review-filter", action="store_true",
                        help="Ignore stitched_review.json and use every training video")
    parser.add_argument("--hunyuan-checkpoint", default=os.environ.get("MODEL_PATH", ""),
                        help="HunyuanVideo-1.5 snapshot dir (default: $MODEL_PATH)")
    args = parser.parse_args()
    assert os.path.isdir(args.hunyuan_checkpoint), \
        "Set --hunyuan-checkpoint or $MODEL_PATH to the HunyuanVideo-1.5 snapshot dir"

    if args.scenes:
        assert args.curation_root, "--scenes needs --curation-root"
        with open(args.scenes) as f:
            scenes = [(os.path.join(args.curation_root, e["output"]), e.get("prompt", ""),
                       os.path.join(args.output_dir, e["output"])) for e in json.load(f)]
    else:
        assert args.scene_dir and args.prompt is not None, "give --scene-dir and --prompt, or --scenes"
        scenes = [(args.scene_dir, args.prompt, args.output_dir)]

    encoder = SceneEncoder(args.hunyuan_checkpoint)
    poses = load_poses(args.poses)
    for scene_dir, prompt, output_dir in scenes:
        if not os.path.isdir(os.path.join(scene_dir, "train")):
            print(f"[SKIP] {scene_dir}: not curated")
            continue
        n = prepare_scene(encoder, scene_dir, poses, prompt, output_dir,
                          image_path=args.image_path, video_name=args.video_name,
                          passing_list=args.passing_list, review_filter=not args.no_review_filter)
        print(f"Done: {n} training trajectories in {output_dir}")


if __name__ == "__main__":
    main()
