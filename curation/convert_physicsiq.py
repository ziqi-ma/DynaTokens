#!/usr/bin/env python3
"""Convert Physics-IQ descriptions.csv to standard pipeline instances JSON.

Physics-IQ stores per-scenario metadata in `descriptions.csv`:

    scenario,description,category,generated_video_name
    0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4,"... Static shot with no camera movement.",Solid Mechanics,0001_perspective-left_trimmed-ball-and-block-fall.mp4

Only take-1 rows are used for model input (take-2 exists for variance baselines).
For each take-1 row we emit one pipeline instance whose initial frame is the
corresponding switch-frame JPG.

Output schema (the curation scenes JSON, as convert_worldscore.py):

    {
      "image":  "<abs path to .../switch-frames/<id>_<perspective>_<name>.jpg>",
      "prompt": "<description with trailing 'Static shot...' sentence stripped>",
      "output": "physicsiq/<category_slug>/<id>_<perspective>",
      "name":   "<id>_<perspective>_<name>",
    }

The "Static shot..." clause is stripped because downstream rollouts synthesize
non-static camera trajectories; leaving the clause in biases HYWP toward
static output and defeats the training goal.

Usage:
    python curation/convert_physicsiq.py \
        --descriptions_csv   datasets/physics-IQ-benchmark/descriptions/descriptions.csv \
        --switch_frames_dir  datasets/PhysicsIQ-Dataset/switch-frames \
        --output_json        physicsiq_scenes.json \
        [--max_instances N]
"""

import argparse
import csv
import json
import os
import re

CATEGORY_SLUG = {
    "Solid Mechanics":  "solid_mechanics",
    "Fluid Dynamics":   "fluid_dynamics",
    "Optics":           "optics",
    "Thermodynamics":   "thermodynamics",
    "Magnetism":        "magnetism",
}

STATIC_SHOT_RE = re.compile(r"\s*Static shot[^.]*\.\s*$", re.IGNORECASE)


def clean_prompt(description: str) -> str:
    """Strip the trailing 'Static shot with no camera movement.' sentence."""
    return STATIC_SHOT_RE.sub("", description.strip()).strip()


def parse_scenario(scenario_field: str) -> tuple[str, str, str]:
    """Parse '0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4'
    into (id='0001', perspective='perspective-left', name_suffix='trimmed-ball-and-block-fall').
    """
    stem = os.path.splitext(scenario_field)[0]
    parts = stem.split("_")
    # Expect: <id>_<perspective-*>_take-<N>_<name...>
    assert len(parts) >= 4, f"Unexpected scenario format: {scenario_field}"
    scen_id     = parts[0]
    perspective = parts[1]
    # parts[2] is 'take-1' / 'take-2' — discard. Rest is the name.
    name_suffix = "_".join(parts[3:])
    return scen_id, perspective, name_suffix


def convert(csv_rows, switch_frames_dir: str) -> list[dict]:
    instances = []
    for row in csv_rows:
        scenario = row["scenario"]
        if "_take-1_" not in scenario:
            continue  # skip take-2

        scen_id, perspective, name_suffix = parse_scenario(scenario)
        category = row["category"].strip()
        slug = CATEGORY_SLUG.get(category)
        assert slug is not None, f"Unknown category '{category}' for {scenario}"

        # Switch-frame naming in the GCS bucket splices "switch-frames_anyFPS"
        # between the ID and the perspective tag, e.g.
        #   0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg
        frame_name = f"{scen_id}_switch-frames_anyFPS_{perspective}_{name_suffix}.jpg"
        image_path = os.path.abspath(os.path.join(switch_frames_dir, frame_name))

        instance_name = f"{scen_id}_{perspective}_{name_suffix}"
        instances.append({
            "image":  image_path,
            "prompt": clean_prompt(row["description"]),
            "output": f"physicsiq/{slug}/{scen_id}_{perspective}",
            "name":   instance_name,
        })
    return instances


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--descriptions_csv", required=True,
                        help="Path to physics-IQ descriptions.csv")
    parser.add_argument("--switch_frames_dir", required=True,
                        help="Path to physics-IQ switch-frames directory")
    parser.add_argument("--output_json", required=True,
                        help="Output path for standard pipeline JSON")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Only convert first N take-1 instances")
    parser.add_argument("--skip_missing_images", action="store_true",
                        help="Drop entries whose switch-frame JPG is not on disk")
    args = parser.parse_args()

    with open(args.descriptions_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    instances = convert(rows, args.switch_frames_dir)

    if args.skip_missing_images:
        before = len(instances)
        instances = [i for i in instances if os.path.isfile(i["image"])]
        dropped = before - len(instances)
        if dropped:
            print(f"Dropped {dropped} instances with missing switch-frames")

    if args.max_instances:
        instances = instances[: args.max_instances]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(instances, f, indent=2)

    # Summary
    by_cat = {}
    for i in instances:
        cat = i["output"].split("/")[1]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"Converted {len(instances)} take-1 instances -> {args.output_json}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
