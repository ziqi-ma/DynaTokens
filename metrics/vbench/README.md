# VBench-2.0 metrics

VBench-2.0 is a text-to-video benchmark with VLM judges. While VBench-2.0 holistically evaluates video models, we use its dynamics-related categories: Dynamic Spatial Relationship (DSR, e.g. "a dog is on the left of the table, then the dog runs to the front of the table") and Motion Order Understanding (MOU, e.g. "a person is cooking, then they suddenly start organizing the pantry"). The evaluation is VBench-2.0, ported as a submodule in `third_party/VBench`.

- DSR: while the original T2V evaluation evaluates both the initial and the final frame, since we pass in the initial frame as conditioning, it is correct by construction. Thus we only evaluate the final frame. Additionally, since VBench does not consider moving cameras, it assumes the main object is always in frame. Given that we allow camera movement and it is possible for the camera to move away from the main object, we add a Gemini evaluation of object presence before applying the object relation evaluation, and report DSR given object presence. We use Gemini-flash-2.5.
- MOU: directly follows VBench-2.0.

Scenes must use VBench-2.0 prompts, since questions and ground truth are looked up by prompt. The prompt of each scene is read from `scene.json` in the rendered folder (written by `dynatokens/inference.sh`), or can be given with `--prompt`.

## Setup

```bash
bash metrics/vbench/setup_env.sh      # conda env `vbench2` + LLaVA-Video and Qwen2.5 judges
```

## Run

```bash
conda activate vbench2
python metrics/vbench/eval_vbench2.py outputs/runs/<run>/eval/ckpt<step> [more rendered folders ...]
```

It prints, for each folder, DSR given object presence and MOU (fraction of videos judged correct).
