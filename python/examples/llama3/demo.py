"""Llama 3.2 1B text generation with pyvai's built-in models.

python python/examples/llama3/demo.py \\
    --weights_dir    data/llama3/ckpts/3.2-1B \\
    --tokenizer_path data/llama3/tokenizer.json

chat.py in this folder is the multi-turn session.
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import Llama3


# The prompt of the C++ demo (examples/llama3/demo.cpp).
PROMPT = (
    "In recent decades, artificial intelligence has progressed from a niche "
    "academic curiosity to a force that quietly reshapes nearly every corner of "
    "modern life. "
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama 3.2 1B generation with pyvai (built-in models)."
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder of the HuggingFace Llama 3 checkpoint (data/llama3/ckpts/<variant>).",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the HuggingFace tokenizer.json."
    )
    parser.add_argument(
        "--max_runtime_context",
        type=int,
        default=4096,
        help="KV-cache budget in tokens, clamped to the 128K trained "
        "context (all of it is several GB of cache; the "
        "C++ demos run at 4096).",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Number of tokens to generate."
    )
    parser.add_argument(
        "--prefill_chunk",
        type=int,
        default=0,
        help="Split a prompt longer than this many tokens into "
        "several forwards (bounds prefill memory; 0: one "
        "forward for the whole prompt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.weights_dir):
        raise FileNotFoundError(f"weights not found: {args.weights_dir}")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer_path}")

    # w_default_dtype=Dtype.F32 runs the weights in FP32; tagging turns profiling on.
    llama = Llama3(
        w_default_dtype=pyvai.Dtype.F16,
        max_runtime_context=args.max_runtime_context,
        prefill_chunk=args.prefill_chunk,
    )
    llama.set_profile_tags()
    llama.load_weights(args.weights_dir)
    llama.load_tokenizer(args.tokenizer_path)

    print(f'Prompt: "{PROMPT}"')
    print(f"Generating {args.max_new_tokens} tokens...\n")

    # stream() yields the completion as it is sampled (generate(): in one piece).
    print("=== Completion ===")
    for piece in llama.stream(PROMPT, max_new_tokens=args.max_new_tokens):
        print(piece, end="", flush=True)
    print()

    llama.print_profile()
    return 0


if __name__ == "__main__":
    main()
