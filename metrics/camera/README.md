# Camera score

For camera, we calculate an aggregate score over rotation and translation error like WorldScore, leveraging the state-of-the-art camera estimation method ViPE (ported as a submodule in `third_party/vipe`). For each segment of the requested trajectory, the end pose estimated by ViPE is compared with the requested one: rotation error is the geodesic angle between the two rotations, and translation error is the L2 distance between the two translations.

## Setup

```bash
git submodule update --init third_party/vipe
bash metrics/camera/setup.sh          # conda env `vipe` + ViPE checkpoints
```

## Run

```bash
conda activate vipe
# 1. estimate camera paths with ViPE for every rendered video of a run / checkpoint
python metrics/camera/vipe_pose_extraction.py --hywp-runs-root outputs/runs --run <run> --ckpt <step>
# 2. camera score (one row per folder)
python metrics/camera/score_camera_endpoint.py --roots outputs/runs/<run>/eval/ckpt<step>
```

The requested trajectory of each video is read from its `pose_string.txt` (written by `dynatokens/inference.sh`).
