import json

from dynatokens.prepare_training_data import load_poses, pose_num_latents, select_train_trajectories


def test_pose_num_latents_counts_motion_steps_and_initial_frame():
    assert pose_num_latents("w-2, left-3") == 6
    assert pose_num_latents("  ") == 1


def test_load_poses_normalizes_missing_splits_and_rejects_unknown_keys(tmp_path):
    poses_path = tmp_path / "poses.json"
    poses_path.write_text(json.dumps({"train": {"one": "w-2"}}))

    assert load_poses(str(poses_path)) == {"train": {"one": "w-2"}, "test": {}}

    poses_path.write_text(json.dumps({"validation": {}}))
    try:
        load_poses(str(poses_path))
    except AssertionError as error:
        assert "unexpected keys" in str(error)
    else:
        raise AssertionError("unknown pose split was accepted")


def test_select_train_trajectories_applies_review_filter(tmp_path):
    scene_dir = tmp_path / "scene"
    passing = scene_dir / "train" / "passing"
    failing = scene_dir / "train" / "failing"
    passing.mkdir(parents=True)
    failing.mkdir(parents=True)
    (passing / "stitched.mp4").touch()
    (failing / "stitched.mp4").touch()
    (passing / "stitched_review.json").write_text(json.dumps({"passed": True}))
    (failing / "stitched_review.json").write_text(json.dumps({"passed": False}))

    selected = select_train_trajectories(
        str(scene_dir),
        {"passing": "w-2", "failing": "s-2", "missing": "a-2"},
        "stitched.mp4",
        passing_list=None,
        review_filter=True,
    )

    assert selected == [("passing", "w-2", str(passing / "stitched.mp4"))]
