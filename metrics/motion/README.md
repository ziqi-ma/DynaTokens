# Motion accuracy (WorldScore)

`eval_motion.py` scores rendered trajectories with WorldScore's motion accuracy, ported as a submodule in `third_party/WorldScore`. Motion accuracy measures whether the prompted object moves: SEA-RAFT optical flow inside a GroundingDINO + SAM mask of the object, compared with the flow of the background.

WorldScore does not consider a moving camera. Since camera motion moves every pixel, we compare the object's flow with the median background flow (`obj_max - bg_median`) instead of WorldScore's maximum (`obj_max - bg_max`, available with `--motion-accuracy-score max_diff`).

## Setup

Follow WorldScore's evaluation setup in `third_party/WorldScore/README.md`: create its `worldscore`
env (including GroundingDINO, segment_anything and `python -m spacy download en_core_web_sm`)
and download the metric checkpoints into `third_party/WorldScore/worldscore/benchmark/metrics/checkpoints`.
Motion accuracy also needs SEA-RAFT's `Tartan-C-T-TSKH-spring540x960-M.pth` in the same folder, which
WorldScore's list does not include; download it from SEA-RAFT's
[Google Drive](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW).
GroundingDINO needs `transformers<5` (it calls `BertModel.get_head_mask`, removed in transformers 5).

## Run

```bash
conda activate worldscore
python metrics/motion/eval_motion.py outputs/runs/<run>/eval/ckpt<step>
```

The prompt comes from `<eval_dir>/scene.json` (written by `dynatokens/inference.sh`); the moving
object is the prompt's head noun (spaCy), or set it with `--object-override`.
