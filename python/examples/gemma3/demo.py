"""Gemma 3 text generation, and image prompting with the 4B / 12B VLMs.

python python/examples/gemma3/demo.py \\
    --weights_dir    data/gemma3/ckpts/4B \\
    --tokenizer_path data/gemma3/tokenizer.json

python python/examples/gemma3/demo.py --preset 1B \\
    --weights_dir    data/gemma3/ckpts/1B \\
    --tokenizer_path data/gemma3/tokenizer.json

chat.py in this folder is the multi-turn session.
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import Gemma3, Gemma3Config


# A text to continue, or the C++ demo's request about its images
# (models/production/gemma3/demo.cpp).
PROMPT = (
    "In recent decades, artificial intelligence has progressed from a niche "
    "academic curiosity to a force that quietly reshapes nearly every corner of "
    "modern life. "
)
IMAGE_PROMPT = "Summarize the image(s)."
IMAGE_PATHS = ["data/gemma3/captured0.png"]


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 3 generation with pyvai (built-in models).")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder of the HuggingFace Gemma 3 checkpoint (google/gemma-3-1b-it, "
        "-4b-it, ...; data/gemma3/ckpts/<variant>): every *.safetensors "
        "in it is read.",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the checkpoint's tokenizer.json."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="4B",
        choices=Gemma3Config.presets,
        help="Checkpoint variant; 4B and 12B are VLMs and take --image_path.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="*",
        default=None,
        help="Images to ask about (a VLM preset; JPEG, PNG, ...). A VLM preset "
        "defaults to the C++ demo's data/gemma3/captured0.png; the flag "
        "with no path asks for text only.",
    )
    parser.add_argument(
        "--max_runtime_context",
        type=int,
        default=4096,
        help="KV-cache budget in tokens, clamped to the trained "
        "context (the C++ demos run at 4096).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Number of tokens to generate."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_path is None:
        has_vision = Gemma3Config.preset(args.preset).vision is not None
        args.image_path = IMAGE_PATHS if has_vision else []
    for path in (args.weights_dir, args.tokenizer_path, *args.image_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # w_default_dtype=Dtype.F32 runs the weights in FP32; tagging turns profiling on.
    gemma = Gemma3(
        Gemma3Config.preset(args.preset),
        w_default_dtype=pyvai.Dtype.F16,
        max_runtime_context=args.max_runtime_context,
    )
    if args.image_path and not gemma.has_vision:
        raise SystemExit(
            f"--image_path goes with a VLM preset; {args.preset} has no vision encoder"
        )
    gemma.set_profile_tags()
    gemma.load_weights(args.weights_dir)
    gemma.load_tokenizer(args.tokenizer_path)

    images = args.image_path or None
    prompt = PROMPT
    if images:
        for b, path in enumerate(images):
            print(f"Image {b}: {path}")
        # One <start_of_image> per image, ahead of the request, as the C++ demo.
        prompt = "<start_of_image>" * len(images) + IMAGE_PROMPT
    print(f'Prompt: "{prompt}"')
    print(f"Generating up to {args.max_new_tokens} tokens...\n")

    # stream() yields the completion as it is sampled (generate(): in one piece).
    print("=== Completion ===")
    for piece in gemma.stream(prompt, max_new_tokens=args.max_new_tokens, images=images):
        print(piece, end="", flush=True)
    print()

    gemma.print_profile()
    return 0


if __name__ == "__main__":
    main()
