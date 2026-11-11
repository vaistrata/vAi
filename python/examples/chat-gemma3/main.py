"""Gemma 3 multi-turn chat with pyvai's built-in models, as
examples/chat-gemma3/main.cpp. A message to a VLM (4B / 12B) names its images with
<image:file> tags, read from --image_dir.

python python/examples/chat-gemma3/main.py \\
    --weights_dir    data/ckpts/gemma3/google/4B \\
    --tokenizer_path data/tokenizers/gemma3/tokenizer.json
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import Gemma3, Gemma3Config, ContextFullError


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 3 chat with pyvai (built-in models).")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder of the HuggingFace Gemma 3 checkpoint (google/gemma-3-1b-it, "
        "-4b-it, ...; data/ckpts/gemma3/google/<variant>): every *.safetensors "
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
        help="Checkpoint variant; 4B and 12B are VLMs and take <image:file> tags.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
        help="Where the files of <image:file> tags are read from (a VLM preset).",
    )
    parser.add_argument(
        "--max_runtime_context",
        type=int,
        default=4096,
        help="KV-cache budget in tokens, clamped to the trained "
        "context (the C++ demos run at 4096).",
    )
    return parser.parse_args()


def run_chat(gemma: Gemma3, image_dir: str) -> None:
    """Multi-turn chat: the ChatSession keeps the conversation in the KV cache and
    prefills only each turn's new text; an <image:file> tag reads from image_dir."""
    chat = gemma.chat(image_dir=image_dir)

    print("\n=== Chat (blank line to quit) ===")
    while True:
        try:
            user_msg = input("\n> ")
        except EOFError:
            break
        if not user_msg:
            break

        # As the C++ demo: end the chat when the turn no longer fits or the reply
        # has filled the cache, and skip a turn whose images cannot be used.
        print()
        full = False
        try:
            for piece in chat.stream(user_msg):
                print(piece, end="", flush=True)
        except ContextFullError:
            full = True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[{e}]")
            continue
        print()

        if full or chat.context_left <= 0:
            print(f"\n[context full: {gemma.tokens_so_far} of {gemma.context_budget} tokens used]")
            break

    print("\n--- Chat ended ---")


def main():
    args = parse_args()
    for path in (args.weights_dir, args.tokenizer_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # w_default_dtype=Dtype.F32 runs the weights in FP32.
    gemma = Gemma3(
        Gemma3Config.preset(args.preset),
        w_default_dtype=pyvai.Dtype.F16,
        max_runtime_context=args.max_runtime_context,
    )
    gemma.load_weights(args.weights_dir)
    gemma.load_tokenizer(args.tokenizer_path)

    run_chat(gemma, args.image_dir)
    return 0


if __name__ == "__main__":
    main()
