"""Behavioral checks for the released evaluation manifest and pose tools."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metrics/eval_manifest.json"
VERIFY = ROOT / "metrics/verify_eval_manifest.py"
GENERATE = ROOT / "metrics/generate_eval_poses.py"


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def read_manifest():
    assert MANIFEST.is_file(), "evaluation manifest has not been created"
    return json.loads(MANIFEST.read_text())


def all_scenes(data):
    return [
        scene
        for benchmark in data["benchmarks"].values()
        for scene in benchmark["scenes"]
    ] + data["additional_released_scenes"]["scenes"]


def test_manifest_counts_and_preserves_released_prompts():
    data = read_manifest()
    for benchmark, target, verified, pending in (("vbench2", 20, 6, 14), ("worldscore", 15, 5, 10)):
        entry = data["benchmarks"][benchmark]
        assert entry["paper_target_scenes"] == target
        assert entry["trajectories_per_scene"] == 6
        assert len(entry["scenes"]) == verified
        assert entry["pending_maintainer_confirmation"] == pending
        assert data["coverage"][benchmark] == {"paper_target": target, "verified": verified, "pending": pending}
    sweep = next(s for s in data["benchmarks"]["vbench2"]["scenes"] if s["id"] == "vbench_sweep")
    assert sweep["prompt"] == "A person is sitting on the couch, then suddenly they get up and start sweeping the floor.\n"
    bird = next(s for s in data["benchmarks"]["vbench2"]["scenes"] if s["id"] == "vbench_bird")
    assert bird["prompt"] == "A bird is behind a apple, then the bird flies to the right of the apple.\n"
    for scene in data["benchmarks"]["worldscore"]["scenes"] + data["additional_released_scenes"]["scenes"]:
        assert scene["prompt"] is None
        assert scene["prompt_status"] == "not_released_empty_prompt_txt"
    assert len(data["additional_released_scenes"]["scenes"]) == 3


def test_all_released_pose_files_have_valid_matrices_and_hashes():
    scenes = all_scenes(read_manifest())
    assert len({scene["id"] for scene in scenes}) == 14
    trajectories = [pose for scene in scenes for pose in scene["trajectories"]]
    assert len(trajectories) == 72
    for scene in scenes:
        prefix = f"s3://dynatokens/dynatokens_checkpoints/{scene['id']}/"
        assert scene["prompt_source"] == prefix + "prompt.txt"
        prompt_raw = (ROOT / scene["prompt_file"]).read_bytes()
        assert hashlib.sha256(prompt_raw).hexdigest() == scene["prompt_sha256"]
        assert (prompt_raw.decode() if prompt_raw else None) == scene["prompt"]
        assert scene["init_image"] == prefix + "init_image.png"
        assert scene["checkpoint"] == prefix + "diffusion_pytorch_model.safetensors"
        for trajectory in scene["trajectories"]:
            assert trajectory["source"] == prefix + f"sample_poses/{trajectory['name']}/pose.json"
            assert trajectory["split"] == "unknown"
            path = ROOT / trajectory["pose_json"]
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == trajectory["sha256"]
            entries = json.loads(raw)
            assert set(entries) == {str(i) for i in range(len(entries))}
            assert len(entries) == trajectory["frame_count"]
            assert len(entries) > 0 and len(entries) % 4 == 0
            for frame in entries.values():
                for name, size in (("w2c", 4), ("intrinsic", 3)):
                    matrix = np.asarray(frame[name], dtype=float)
                    assert matrix.shape == (size, size)
                    assert np.isfinite(matrix).all()


def test_generator_reproduces_released_right7up8(tmp_path):
    result = run_script(GENERATE, "generate", "--pose", "right-7,up-8", "--out-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pose_string.txt").read_text() == "right-7,up-8\n"
    actual = json.loads((tmp_path / "pose.json").read_text())
    expected = json.loads((ROOT / "metrics/eval_poses/vbench_bird/right7up8/pose.json").read_text())
    assert actual.keys() == expected.keys()
    for key in expected:
        for field in ("w2c", "intrinsic"):
            np.testing.assert_allclose(actual[key][field], expected[key][field], atol=1e-6, rtol=0)


@pytest.mark.parametrize("pose", ["fly-15", "w-7,up-7", "w-1.5,up-14", "w--1,up-16", "", "w-15,unknown-0"])
def test_generator_rejects_invalid_pose_strings(pose, tmp_path):
    result = run_script(GENERATE, "generate", "--pose", pose, "--out-dir", tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "pose.json").exists()


def test_generator_verify_checks_release():
    result = run_script(GENERATE, "verify", "--manifest", MANIFEST)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_verifier_accepts_real_manifest():
    result = run_script(VERIFY, "--manifest", MANIFEST)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PASS")
    assert len(result.stdout.strip().splitlines()) == 1


@pytest.mark.parametrize("defect", ["sha256", "coverage", "duplicate", "prompt", "split"])
def test_verifier_rejects_broken_manifest(defect, tmp_path):
    data = read_manifest()
    if defect == "sha256":
        data["benchmarks"]["vbench2"]["scenes"][0]["trajectories"][0]["sha256"] = "0" * 64
    elif defect == "coverage":
        data["benchmarks"]["vbench2"]["pending_maintainer_confirmation"] = 13
    elif defect == "duplicate":
        data["additional_released_scenes"]["scenes"][0]["id"] = data["benchmarks"]["vbench2"]["scenes"][0]["id"]
    elif defect == "prompt":
        data["benchmarks"]["worldscore"]["scenes"][0]["prompt_status"] = "available"
    else:
        data["benchmarks"]["vbench2"]["scenes"][0]["trajectories"][0]["split"] = "seen"
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data))
    result = run_script(VERIFY, "--manifest", broken)
    assert result.returncode != 0


@pytest.mark.parametrize("defect", ["missing_key", "bad_matrix", "nonfinite", "bad_frame_count"])
def test_verifier_rejects_invalid_pose_even_with_matching_hash(defect, tmp_path):
    data = read_manifest()
    entry = data["benchmarks"]["vbench2"]["scenes"][0]["trajectories"][0]
    root = tmp_path / "repo"
    (root / "metrics").mkdir(parents=True)
    shutil.copytree(ROOT / "metrics/eval_poses", root / "metrics/eval_poses")
    shutil.copy2(VERIFY, root / "metrics/verify_eval_manifest.py")
    path = root / entry["pose_json"]
    pose = json.loads(path.read_text())
    if defect == "missing_key":
        pose["99"] = pose.pop("1")
    elif defect == "bad_matrix":
        pose["0"]["w2c"] = [[1]]
    elif defect == "nonfinite":
        pose["0"]["intrinsic"][0][0] = float("nan")
    else:
        pose.pop(str(len(pose) - 1))
        entry["frame_count"] = len(pose)
    raw = json.dumps(pose).encode()
    path.write_bytes(raw)
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest = root / "metrics/eval_manifest.json"
    manifest.write_text(json.dumps(data))
    result = run_script(root / "metrics/verify_eval_manifest.py", "--manifest", manifest)
    assert result.returncode != 0
