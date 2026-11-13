"""PaliGemma: one prompt about one image, with pyvai's built-in models.

python python/examples/paligemma/demo.py \\
    --weights_dir    data/paligemma/ckpts/3B-224 \\
    --tokenizer_path data/paligemma/tokenizer.json \\
    --image_path     data/paligemma/dog.jpg

Mirrors examples/paligemma/demo.cpp: the image is prefilled with the prompt,
then the answer is decoded one token at a time.
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import PaliGemma, PaliGemmaConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaliGemma image prompting with pyvai (built-in models)."
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder holding the HuggingFace PaliGemma *.safetensors shards "
        "(data/paligemma/ckpts/<variant>; "
        "google/paligemma-3b-mix-224 or -448).",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the HuggingFace tokenizer.json."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Image to ask about (JPEG, PNG, ...)."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="3B-224",
        choices=PaliGemmaConfig.presets,
        help="Checkpoint preset; picks the image size.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="caption en",
        help='Task prefix ("caption en", "answer en <question>", ...).',
    )
    parser.add_argument(
        "--max_runtime_context",
        type=int,
        default=2048,
        help="KV-cache budget in tokens (the C++ demo runs at 2048); "
        "an image costs num_patches of it.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Cap on the answer length.")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.weights_dir, args.tokenizer_path, args.image_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    model = PaliGemma(
        PaliGemmaConfig.preset(args.preset),
        w_default_dtype=pyvai.Dtype.F16,
        max_runtime_context=args.max_runtime_context,
    )
    model.set_profile_tags()
    model.load_weights(args.weights_dir)
    model.load_tokenizer(args.tokenizer_path)

    print(f"Image: {args.image_path}, squashed to {model.image_size}x{model.image_size}")
    print(f'Prompt: "{args.prompt}"\n')

    print("=== Answer ===")
    for piece in model.stream(
        args.prompt, max_new_tokens=args.max_new_tokens, image=args.image_path
    ):
        print(piece, end="", flush=True)
    print()

    model.print_profile()
    return 0


if __name__ == "__main__":
    main()
