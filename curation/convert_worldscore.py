#!/usr/bin/env python3
"""Convert WorldScore dynamic JSON to standard pipeline instances JSON.

Reads the WorldScore format:
    {"image": "./dynamic/...", "prompt": "...", "visual_style": "rigid", "motion_type": "014", ...}

Writes the standard pipeline format:
    {"image": "dynamic/.../image.png", "prompt": "...", "output": "dynamic/rigid/014/name", "name": "name"}

Usage:
    python curation/convert_worldscore.py \
        --dynamic_json datasets/WorldScore-Dataset/dynamic/dynamic_test.json \
        --dataset_root datasets/WorldScore-Dataset \
        --output_json  worldscore_scenes.json \
        [--max_instances N]
"""

import argparse
import json
import os


def convert_worldscore(ws_instances, dataset_root):
    """Convert WorldScore JSON entries to standard pipeline format."""
    pipeline_instances = []
    for ws in ws_instances:
        name = os.path.splitext(os.path.basename(ws["image"]))[0]
        # Resolve image to absolute path
        image_path = os.path.join(dataset_root, ws["image"].lstrip("./"))
        pipeline_instances.append({
            "image": os.path.abspath(image_path),
            "prompt": ws["prompt"],
            "output": f"dynamic/{ws['visual_style']}/{ws['motion_type']}/{name}",
            "name": name,
        })
    return pipeline_instances


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dynamic_json", required=True,
                        help="Path to WorldScore dynamic.json")
    parser.add_argument("--dataset_root", required=True,
                        help="Path to WorldScore-Dataset root")
    parser.add_argument("--output_json", required=True,
                        help="Output path for standard pipeline JSON")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Only convert first N instances")
    args = parser.parse_args()

    with open(args.dynamic_json) as f:
        ws_instances = json.load(f)

    if args.max_instances:
        ws_instances = ws_instances[:args.max_instances]

    pipeline_instances = convert_worldscore(ws_instances, args.dataset_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(pipeline_instances, f, indent=2)

    print(f"Converted {len(pipeline_instances)} instances -> {args.output_json}")


if __name__ == "__main__":
    main()
