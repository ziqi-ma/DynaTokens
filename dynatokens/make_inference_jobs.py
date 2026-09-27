"""Build a jobs JSON for scripts/inference.py (used by inference.sh) from a prepared scene.

  held-out trajectories  <data-base>/test/<name>/pose.json   -> <out-dir>/unseen_<name>/
      rendered from the scene's first frame
  training trajectories  <data-base>/train/<name>/pose.json  -> <out-dir>/seen_<name>/   (with --seen)
      rendered from the curated video's first frame; the video is the GT for a frame-MSE metric

Each output dir also gets pose_string.txt (read by metrics/camera), and <out-dir> gets a
copy of scene.json (prompt, read by metrics/vbench and metrics/motion). Trajectories whose
gen.mp4 already exists are skipped.
"""
import argparse
import glob
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-base", required=True, help="Prepared scene dir (from prepare_training_data.py)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs-json", required=True, help="Where to write the jobs list")
    parser.add_argument("--seen", action="store_true", help="Also render the training trajectories")
    parser.add_argument("--names", nargs="*", help="Only these trajectory names")
    parser.add_argument("--prompt", default="", help="Text prompt for generation (default: none)")
    args = parser.parse_args()

    with open(os.path.join(args.data_base, "scene.json")) as f:
        scene = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(os.path.join(args.data_base, "scene.json"), args.out_dir)  # prompt etc. for the metrics
    splits = [("test", "unseen")] + ([("train", "seen")] if args.seen else [])
    jobs = []
    for split, tag in splits:
        for pose_json in sorted(glob.glob(os.path.join(args.data_base, split, "*", "pose.json"))):
            name = os.path.basename(os.path.dirname(pose_json))
            if args.names and name not in args.names:
                continue
            out = os.path.abspath(os.path.join(args.out_dir, f"{tag}_{name}"))
            if os.path.isfile(os.path.join(out, "gen.mp4")):
                continue
            image = scene["image_path"]
            if split == "train":
                gt = os.path.join(scene["scene_dir"], "train", name, scene.get("video_name", "stitched.mp4"))
                image = gt if os.path.isfile(gt) else image
            os.makedirs(out, exist_ok=True)
            pose_string = os.path.join(os.path.dirname(pose_json), "pose_string.txt")
            if os.path.isfile(pose_string):
                shutil.copy(pose_string, out)
            jobs.append({"pose_json": os.path.abspath(pose_json), "image_path": image,
                         "output_dir": out, "prompt": args.prompt})
    with open(args.jobs_json, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"{len(jobs)} trajectories to render -> {args.out_dir}")


if __name__ == "__main__":
    main()
