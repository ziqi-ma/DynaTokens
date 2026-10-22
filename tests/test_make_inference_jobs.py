import json

from dynatokens import make_inference_jobs


def test_main_creates_unseen_jobs_and_copies_scene_metadata(tmp_path, monkeypatch):
    data_base = tmp_path / "training_data"
    test_pose = data_base / "test" / "orbit"
    test_pose.mkdir(parents=True)
    (data_base / "scene.json").write_text(json.dumps({"image_path": "init.png", "prompt": "a scene"}))
    (test_pose / "pose.json").write_text("{}")
    (test_pose / "pose_string.txt").write_text("left-2\n")

    output_dir = tmp_path / "renders"
    jobs_path = output_dir / "jobs.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_inference_jobs.py",
            "--data-base", str(data_base),
            "--out-dir", str(output_dir),
            "--jobs-json", str(jobs_path),
            "--prompt", "a prompt",
        ],
    )

    make_inference_jobs.main()

    jobs = json.loads(jobs_path.read_text())
    assert len(jobs) == 1
    assert jobs[0]["output_dir"].endswith("unseen_orbit")
    assert jobs[0]["image_path"] == "init.png"
    assert (output_dir / "scene.json").read_text() == (data_base / "scene.json").read_text()
    assert (output_dir / "unseen_orbit" / "pose_string.txt").read_text() == "left-2\n"


def test_main_skips_existing_outputs_and_filters_names(tmp_path, monkeypatch):
    data_base = tmp_path / "training_data"
    for name in ("keep", "skip"):
        pose_dir = data_base / "test" / name
        pose_dir.mkdir(parents=True)
        (pose_dir / "pose.json").write_text("{}")
    (data_base / "scene.json").write_text(json.dumps({"image_path": "init.png"}))
    output_dir = tmp_path / "renders"
    existing = output_dir / "unseen_keep"
    existing.mkdir(parents=True)
    (existing / "gen.mp4").touch()
    jobs_path = output_dir / "jobs.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_inference_jobs.py",
            "--data-base", str(data_base),
            "--out-dir", str(output_dir),
            "--jobs-json", str(jobs_path),
            "--names", "keep", "skip",
        ],
    )

    make_inference_jobs.main()

    jobs = json.loads(jobs_path.read_text())
    assert len(jobs) == 1
    assert jobs[0]["output_dir"].endswith("unseen_skip")
