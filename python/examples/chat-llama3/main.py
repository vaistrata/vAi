"""Llama 3.2 1B multi-turn chat with pyvai's built-in models, as
examples/chat-llama3/main.cpp.

python python/examples/chat-llama3/main.py \\
    --weights_dir    data/ckpts/llama3/meta/3.2-1B \\
    --tokenizer_path data/tokenizers/llama3/tokenizer.json
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import Llama3, ContextFullError


def parse_args():
    parser = argparse.ArgumentParser(description="Llama 3.2 1B chat with pyvai (built-in models).")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder of the HuggingFace Llama 3 checkpoint (data/ckpts/llama3/meta/<variant>).",
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
        "--prefill_chunk",
        type=int,
        default=0,
        help="Split a prompt longer than this many tokens into "
        "several forwards (bounds prefill memory; 0: one "
        "forward for the whole prompt).",
    )
    return parser.parse_args()


def run_chat(llama: Llama3) -> None:
    """Multi-turn chat: the ChatSession keeps the conversation in the KV cache and
    prefills only each turn's new text."""
    chat = llama.chat()

    print("\n=== Chat (blank line to quit) ===")
    while True:
        try:
            user_msg = input("\n> ")
        except EOFError:
            break
        if not user_msg:
            break

        # As the C++ demo: end the chat when the turn no longer fits or the reply
        # has filled the cache.
        print()
        full = False
        try:
            for piece in chat.stream(user_msg):
                print(piece, end="", flush=True)
        except ContextFullError:
            full = True
        print()

        if full or chat.context_left <= 0:
            print(f"\n[context full: {llama.tokens_so_far} of {llama.context_budget} tokens used]")
            break

    print("\n--- Chat ended ---")


def main():
    args = parse_args()
    if not os.path.exists(args.weights_dir):
        raise FileNotFoundError(f"weights not found: {args.weights_dir}")
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer_path}")

    # w_default_dtype=Dtype.F32 runs the weights in FP32.
    llama = Llama3(
        w_default_dtype=pyvai.Dtype.F16,
        max_runtime_context=args.max_runtime_context,
        prefill_chunk=args.prefill_chunk,
    )
    llama.load_weights(args.weights_dir)
    llama.load_tokenizer(args.tokenizer_path)

    run_chat(llama)
    return 0


if __name__ == "__main__":
    main()
