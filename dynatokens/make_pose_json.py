"""Write a pose.json (the camera-trajectory format used for training and inference)
from a WASD pose string, e.g. to render a new trajectory with inference.sh.

  PYTHONPATH=model/hywp python dynatokens/make_pose_json.py --pose "<pose string>" --out-dir my_pose/
  -> my_pose/pose.json, my_pose/pose_string.txt

Pose string: comma-separated segments "<action>-<latents>", actions w/s/a/d (translate)
and up/down/left/right (rotate). Use the same total length as the training trajectories.
"""
import argparse

from prepare_training_data import write_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose", required=True, help="Pose string")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(f"{args.pose!r} -> {write_pose(args.out_dir, args.pose)}")


if __name__ == "__main__":
    main()
