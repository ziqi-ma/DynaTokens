#!/usr/bin/env python3
"""De-blur state images using Gemini image editing.

Applies motion-blur removal to state_1..N images in a states/ directory,
skipping state_0 (original image, no blur expected).
Writes state_i_deblurred.png next to each blurred state_i.png (used by
hywp_rerender.py when present). Uses a marker file to skip if already done.

Usage:
    python deblur_keyframes.py --states-dir path/to/states --n-steps 2
"""

import argparse
import os
from pathlib import Path

from google import genai
from google.genai import types

BLUR_CHECK_PROMPT = (
    "Does this image have motion blur or camera shake blur? "
    "Answer with just YES or NO."
)

DEBLUR_INSTRUCTION = (
    "This image has motion blur. Sharpen the image to remove all motion blur "
    "and make the subject crisp and clear. The subject, colors, background, "
    "and composition must remain identical — do not change anything except to "
    "remove the blur and restore sharpness."
)

MARKER = "deblur_done.txt"


def has_motion_blur(client, image_path):
    """Ask Gemini whether the image has motion blur. Returns True if YES."""
    p = Path(image_path)
    image_bytes = p.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[image_part, BLUR_CHECK_PROMPT],
    )
    answer = (getattr(response, "text", "") or "").strip().upper()
    return answer.startswith("YES")


def deblur_image(client, image_path):
    """De-blur image and save as state_N_deblurred.png alongside the original."""
    p = Path(image_path)
    out = p.with_stem(p.stem + "_deblurred")

    image_bytes = p.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[image_part, DEBLUR_INSTRUCTION],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            part.as_image().save(out)
            return

    raise RuntimeError(f"No image returned for {image_path}. Text: {response.text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states-dir", required=True,
                        help="Path to states/ directory containing state_0.png .. state_N.png")
    parser.add_argument("--n-steps", type=int, required=True,
                        help="Number of state steps (de-blurs state_1 .. state_N)")
    args = parser.parse_args()

    states_dir = Path(args.states_dir)
    marker = states_dir / MARKER
    if marker.exists():
        print(f"  De-blur already done ({marker}), skipping.")
        return

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    targets = [states_dir / f"state_{i}.png" for i in range(1, args.n_steps + 1)]
    targets = [p for p in targets if p.exists()]

    if not targets:
        print(f"  No state images to de-blur in {states_dir}")
        marker.write_text("no targets")
        return

    print(f"  Checking {len(targets)} state image(s) for motion blur ...")
    for p in targets:
        print(f"    {p.name}: checking ...")
        if has_motion_blur(client, p):
            print(f"    {p.name}: blur detected, de-blurring ...")
            deblur_image(client, p)
            print(f"    {p.name}: done.")
        else:
            print(f"    {p.name}: no blur, skipping.")

    marker.write_text("done")
    print(f"  De-blur complete.")


if __name__ == "__main__":
    main()
