"""
VBench-2.0 scores for rendered trajectories on Dynamic Spatial Relationship (DSR) and
Motion Order Understanding (MOU), see metrics/vbench/README.md.

Each EVAL_DIR is one scene: <eval_dir>/*/gen.mp4 (from dynatokens/inference.sh), with its
prompt in <eval_dir>/scene.json (or --prompt). The prompt must be a VBench-2.0 prompt.

Usage (vbench2 env, see metrics/vbench/setup_env.sh):
    python metrics/vbench/eval_vbench2.py outputs/runs/<run>/eval/ckpt<step> [more eval dirs ...]
"""
import argparse
import copy
import glob
import json
import os
import re
import sys
import tempfile

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VBENCH2_DIR = os.path.join(REPO_ROOT, "third_party", "VBench", "VBench-2.0")
sys.path.insert(0, VBENCH2_DIR)

from vbench2 import dynamic_spatial_relationship as dsr  # noqa: E402
from vbench2.motion_order_understanding import compute_motion_order_understanding  # noqa: E402
from vbench2.utils import CACHE_DIR  # noqa: E402

DIMENSIONS = ["Dynamic_Spatial_Relationship", "Motion_Order_Understanding"]
SUBMODULES = {"llava": os.path.join(CACHE_DIR, "lmms-lab/LLaVA-Video-7B-Qwen2"),
              "qwen": os.path.join(CACHE_DIR, "Qwen/Qwen2.5-7B-Instruct")}


def final_position_llava(prompt_dict_ls, model, tokenizer, image_processor, device):
    """Upstream dsr.LLaVA_Video, asking only the final-position question (base_question[1])."""
    processed_json = []
    for prompt_dict in prompt_dict_ls:
        question = prompt_dict["auxiliary_info"][1]
        for video_path in prompt_dict["video_list"]:
            video, _, _ = dsr.load_video(video_path, 64, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            conv = copy.deepcopy(dsr.conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], dsr.DEFAULT_IMAGE_TOKEN +
                                f"Please answer yes or no only for the following questions.\n{question}")
            conv.append_message(conv.roles[1], None)
            input_ids = dsr.tokenizer_image_token(conv.get_prompt(), tokenizer, dsr.IMAGE_TOKEN_INDEX,
                                                  return_tensors="pt").unsqueeze(0).to(device)
            cont = model.generate(input_ids, images=[video[-1].unsqueeze(0)], modalities=["image"],
                                  do_sample=False, temperature=0, max_new_tokens=4096)
            answer = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            processed_json.append({"video_path": video_path, "video_results": int("yes" in answer.lower())})
    return None, processed_json


# ── Gemini judges (DSR final position, object presence), as used for the reported results ──
GEMINI_MODEL = "gemini-2.5-flash"
_DIRECTIONS = {"right", "left", "front", "back", "top", "bottom", "side"}


def last_frame_jpeg(p):
    import io
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(p, ctx=cpu(0), num_threads=1)
    buf = io.BytesIO()
    Image.fromarray(vr[-1].asnumpy()).save(buf, format="JPEG")
    return buf.getvalue()


def ask_gemini(prompt, jpeg):
    import base64
    import time
    import google.generativeai as genai
    client = genai.GenerativeModel(GEMINI_MODEL)
    img = {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}
    for attempt in range(3):
        try:
            return client.generate_content([prompt, img]).text.strip()
        except Exception as e:
            print(f"  gemini error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return "error"


def _objects_from_question(q):
    """'Is the kangaroo on the right of the box? (yes or no)' -> ['kangaroo', 'box']"""
    nouns = [w.lower() for w in re.findall(r"\bthe (\w+)\b", q, re.I)]
    out = []
    for n in nouns:
        if n in _DIRECTIONS or n in out:
            continue
        out.append(n)
        if len(out) == 2:
            break
    return out


def object_presence(prompt_dict_ls):
    """{video_path: 1/0} — is the main object (DSR subject) in the last frame?
    Videos with a Gemini error are left out."""
    present = {}
    for pd in prompt_dict_ls:
        objs = _objects_from_question(pd["auxiliary_info"][1])
        if not objs:
            continue
        prompt = f"Answer only 'yes' or 'no': is there a {objs[0]} in the frame?"
        for vp in pd["video_list"]:
            ans = ask_gemini(prompt, last_frame_jpeg(vp))
            if ans != "error":
                present[vp] = int("yes" in ans.lower())
    return present


def final_position_gemini(prompt_dict_ls):
    """DSR final-position question on the last frame, answered by Gemini."""
    processed_json = []
    for pd in prompt_dict_ls:
        q = pd["auxiliary_info"][1].replace("on the front of", "in front of")
        prompt = f"Look at this image carefully. Answer only 'yes' or 'no' to the following question:\n{q}"
        for vp in pd["video_list"]:
            ans = ask_gemini(prompt, last_frame_jpeg(vp))
            processed_json.append({"video_path": vp, "video_results": 1 if "yes" in ans.lower() else 0})
    return processed_json


def vbench_info(prompt, dimension, videos):
    """VBench-2.0 info entry for this prompt / dimension, or None if the prompt is not in it."""
    for entry in json.load(open(os.path.join(VBENCH2_DIR, "vbench2", "VBench2_full_info.json"))):
        if entry["prompt_en"] == prompt and entry["dimension"][0] == dimension:
            return [{**entry, "dimension": [dimension], "video_list": videos}]
    return None


def use_sdpa_single_gpu():
    """Load LLaVA with sdpa attention and the LLaVA / Qwen judges on a single GPU, as for the
    reported results (upstream VBench-2.0 uses flash_attention_2 with device_map="auto")."""
    from transformers import AutoModelForCausalLM
    from vbench2 import motion_order_understanding as mou

    load_llava = mou.load_pretrained_model

    def load_llava_sdpa(*a, **kw):
        kw.setdefault("attn_implementation", "sdpa")
        kw["device_map"] = {"": "cuda:0"}
        return load_llava(*a, **kw)
    mou.load_pretrained_model = dsr.load_pretrained_model = load_llava_sdpa

    from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

    @classmethod
    def from_pretrained_single_gpu(cls, *a, **kw):
        if kw.get("device_map") == "auto":
            kw["device_map"] = {"": "cuda:0"}
        return from_pretrained(cls, *a, **kw)
    AutoModelForCausalLM.from_pretrained = from_pretrained_single_gpu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dirs", nargs="+", help="Eval dirs, one per scene")
    ap.add_argument("--prompt", help="Scene prompt (default: <eval_dir>/scene.json)")
    ap.add_argument("--dims", nargs="+", default=DIMENSIONS, choices=DIMENSIONS)
    ap.add_argument("--dsr-judge", default="gemini", choices=["gemini", "llava"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2"],
                    help="LLaVA attention: sdpa on a single GPU (as for the reported results), or "
                         "upstream VBench-2.0's flash_attention_2 with device_map=auto (needs flash-attn)")
    args = ap.parse_args()

    if args.attn == "sdpa":
        use_sdpa_single_gpu()
    dsr.LLaVA_Video = final_position_llava
    if "Dynamic_Spatial_Relationship" in args.dims:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    device = torch.device("cuda")
    eval_dirs = [os.path.abspath(d) for d in args.eval_dirs]
    os.chdir(VBENCH2_DIR)
    for eval_dir in eval_dirs:
        prompt = args.prompt or json.load(open(os.path.join(eval_dir, "scene.json")))["prompt"]
        videos = sorted(glob.glob(os.path.join(eval_dir, "*", "gen.mp4")))
        scores = {}
        for dim in args.dims:
            info = vbench_info(prompt, dim, videos)
            if info is None:
                print(f"{eval_dir}: not a VBench-2.0 {dim} prompt, skipping")
                continue
            if dim == "Dynamic_Spatial_Relationship" and args.dsr_judge == "gemini":
                results = final_position_gemini(info)
            else:
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump(info, f)
                compute = (dsr.compute_dynamic_spatial_relationship if dim == "Dynamic_Spatial_Relationship"
                           else compute_motion_order_understanding)
                _, results = compute(f.name, device, SUBMODULES)
                os.remove(f.name)
            if dim == "Dynamic_Spatial_Relationship":
                # DSR is reported given object presence
                present = object_presence(info)
                valid = [r["video_results"] for r in results if present.get(r["video_path"]) == 1]
                scores["DSR given object presence"] = sum(valid) / len(valid) if valid else float("nan")
            else:
                valid = [r["video_results"] for r in results if r.get("video_results", -1) != -1]
                scores["MOU"] = sum(valid) / len(valid) if valid else float("nan")
        print(eval_dir, scores)


if __name__ == "__main__":
    main()
