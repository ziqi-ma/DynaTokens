# DynaTokens: Teaching Dynamics to Camera-Controlled Video Models at Test Time
**Test-time training to let camera-controlled world models learn dynamics**

### NeurIPS 2026

[Ziqi Ma][zm], [Hongqiao Chen][hc], [Georgia Gkioxari][gg]

[[`Project Page`](https://glab-caltech.github.io/dynatokens/)] [[`arXiv`](https://arxiv.org/abs/2609.35704)]

![teaser](media/teaser.png?raw=true)

## Table of Contents:
1. [Overview](#overview)
2. [Environment Setup](#environment)
3. [Data Curation](#datacuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Evaluation](#evaluation)
7. [Checkpoints](#checkpoints)
8. [Citing](#citing)

## Overview <a name="overview"></a>
Camera-controlled video models handle camera-induced motion well in static scenes, but struggle with scene dynamics: objects stay static, move incorrectly, or degrade. DynaTokens is a lightweight set of learnable, scene-specific tokens that teach dynamics to an existing camera-controlled world model. Since camera motion affects the view globally while object dynamics are spatially localized, the tokens are injected through cross-attention and trained on a few example trajectories of a scene with the base model frozen, enabling dynamics under new camera paths.

## Environment Setup <a name="environment"></a>
Training and inference require GPUs with CUDA support. Data curation, training and inference share one environment; the evaluation metrics have their own (see [Evaluation](#evaluation)).
```
git clone --recursive https://github.com/ziqi-ma/DynaTokens.git
cd DynaTokens
conda env create -f environment.yml
conda activate dynatoken
```
Optionally install FlashAttention
```
pip install flash-attn --no-build-isolation
```

Download the checkpoints of HY-WorldPlay and FLUX.1-Redux (vision encoder; the HuggingFace token needs access to [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev))
```
python model/hywp/download_models.py --hf_token [HuggingFace token]
```

## Data Curation <a name="datacuration"></a>
We curate a few samples per scene at test time, leveraging Gemini and Kling in addition to the base camera-controlled model.

Camera trajectories are always an input, given as a poses JSON file with `train` and `test` trajectories that is used by every step. A pose string is a comma-separated list of segments `<action>-<number-of-latents>`, with `w`/`s`/`a`/`d` (translate) and `up`/`down`/`left`/`right` (rotate). All poses have the same number of segments and the same length, and the number of motion steps + 1 must be a multiple of 4. `train` trajectories supervise the tokens; `test` trajectories are new camera paths to render.

Example command for your own scenes (a JSON list of `{"image", "prompt", "output"}`)
```
INSTANCES_JSON=[scenes json] POSES_JSON=[poses json] OUTPUT_ROOT=outputs/curation/[set] N_STEPS=2 \
    bash curation/run.sh
```
Details of every step are in [`curation/README.md`](curation/README.md).

## Training <a name="training"></a>
First encode the supervision videos of a scene
```
PYTHONPATH=model/hywp python dynatokens/prepare_training_data.py \
    --scene-dir outputs/curation/[set]/[scene] --poses [poses json] \
    --prompt "[scene prompt]" --output-dir outputs/training_data/[scene]
```
To prepare every scene of a curation run at once, pass the same scenes JSON instead: `--scenes [scenes json] --curation-root outputs/curation/[set]`.

Then train the temporal tokens
```
DATA_BASE=outputs/training_data/[scene] bash dynatokens/train.sh
```
By default this trains for 3000 steps with learning rate 3e-3, weight decay 1e-5 and token dimension 64 on 4 GPUs, saving checkpoints every 500 steps to `outputs/runs/[run]/checkpoint-[step]`. In practice, some scenes might converge in fewer steps. All options are listed at the top of [`dynatokens/train.sh`](dynatokens/train.sh).

## Inference <a name="inference"></a>
Download a released checkpoint (see [Checkpoints](#checkpoints)) and render sample trajectories
```
CKPT=checkpoints/vbench_sweep bash dynatokens/inference.sh
```
For every `sample_poses/[pose]/pose.json` in the checkpoint folder, the script renders that camera trajectory starting from `init_image.png`, conditioned on the text prompt in `prompt.txt`. To render a new trajectory, add a folder with its `pose.json` under `sample_poses/`; `dynatokens/make_pose_json.py` writes one from a pose string.

## Evaluation <a name="evaluation"></a>
Our evaluation metrics include VBench dynamic spatial relation, motion ordering, WorldScore motion score, and ViPE camera score. First render the `test` trajectories of a trained scene
```
CKPT=outputs/runs/[run]/checkpoint-[step] DATA_BASE=outputs/training_data/[scene] bash dynatokens/inference.sh
```
which writes `outputs/runs/[run]/eval/ckpt[step]/`, then score it.

VBench dynamic spatial relation and motion ordering, see [`metrics/vbench/README.md`](metrics/vbench/README.md)
```
bash metrics/vbench/setup_env.sh && conda activate vbench2
python metrics/vbench/eval_vbench2.py outputs/runs/[run]/eval/ckpt[step]
```
WorldScore motion score, see [`metrics/motion/README.md`](metrics/motion/README.md)
```
conda activate worldscore
python metrics/motion/eval_motion.py outputs/runs/[run]/eval/ckpt[step]
```
Camera score (ViPE), see [`metrics/camera/README.md`](metrics/camera/README.md)
```
conda activate vipe
python metrics/camera/vipe_pose_extraction.py --hywp-runs-root outputs/runs --run [run] --ckpt [step]
python metrics/camera/score_camera_endpoint.py --roots outputs/runs/[run]/eval/ckpt[step]
```

## Checkpoints <a name="checkpoints"></a>
Trained DynaTokens checkpoints are at `s3://dynatokens/dynatokens_checkpoints/`, one folder per scene:
```
aws s3 cp --recursive --no-sign-request s3://dynatokens/dynatokens_checkpoints/ checkpoints/
```

## Citing <a name="citing"></a>
Please use the following BibTeX entry if you find our work helpful!

```BibTex
@inproceedings{ma2026dynatokens,
 title = {DynaTokens: Teaching Dynamics to Camera-Controlled Video Models at Test Time},
 author = {Ma, Ziqi and Chen, Hongqiao and Gkioxari, Georgia},
 booktitle = {Advances in Neural Information Processing Systems},
 year = {2026}
}
```

The code in [`model/hywp/`](model/hywp/) is derived from HY-WorldPlay and is distributed under the [Tencent HY-WorldPlay Community License](model/hywp/License.txt).

[zm]: https://ziqi-ma.github.io/
[hc]: https://www.linkedin.com/in/hongqiao-chen/
[gg]: https://gkioxari.github.io/
