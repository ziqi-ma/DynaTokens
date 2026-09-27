#!/usr/bin/env python3
"""Run camera trajectory eval on a flat baseline / ablation layout:
   <root>/<method>/<instance>/<seen|unseen>_<traj>/gen.mp4

Distributes (method, instance) jobs across GPUs and invokes
vipe_pose_extraction.py with --flat-layout --skip-existing, e.g. to score
DynaToken and baseline methods rendered on the same trajectories.

Usage (inside the vipe conda env):
    conda activate vipe
    python metrics/camera/run_camera_eval_baselines.py \\
        --root outputs/eval/methods --methods dynatoken hywp

    # Custom GPUs
    python metrics/camera/run_camera_eval_baselines.py --gpus 0,1,2,3
"""
import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

SCRIPT = Path(__file__).resolve().parent / "vipe_pose_extraction.py"


def run_eval(root: Path, method: str, instance_dir: Path, gpu: int):
    cmd = [
        sys.executable, str(SCRIPT),
        "--hywp-runs-root", str(root / method),
        "--run", instance_dir.name,
        "--ckpt", "1",
        "--flat-layout",
        "--skip-existing",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    label = f"{method}/{instance_dir.name} (GPU {gpu})"
    print(f"[START] {label}", flush=True)
    r = subprocess.run(cmd, env=env)
    status = "DONE" if r.returncode == 0 else f"FAIL {r.returncode}"
    print(f"[{status}] {label}", flush=True)


def worker(q: Queue, root: Path, gpu: int):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        method, inst = item
        run_eval(root, method, inst, gpu)
        q.task_done()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Layout root: <root>/<method>/<instance>/<seen|unseen>_<traj>/gen.mp4")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Method dir names under <root>.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU indices (default: 0,1,2,3,4,5,6,7).")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    work = []
    for method in args.methods:
        mroot = args.root / method
        if not mroot.is_dir():
            print(f"[WARN] missing {mroot}", flush=True)
            continue
        for inst in sorted(p for p in mroot.iterdir() if p.is_dir() and not p.name.startswith(".")):
            work.append((method, inst))

    print(f"Queuing {len(work)} jobs across GPUs {gpus} (root={args.root})", flush=True)
    queues = [Queue() for _ in gpus]
    threads = []
    for q, g in zip(queues, gpus):
        t = Thread(target=worker, args=(q, args.root, g), daemon=True)
        t.start()
        threads.append(t)
    for item, q in zip(work, itertools.cycle(queues)):
        q.put(item)
    for q in queues:
        q.put(None)
    for t in threads:
        t.join()
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
