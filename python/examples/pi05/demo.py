"""pi0.5: one action chunk from camera views, a task string and the robot's state.

python python/examples/pi05/demo.py \\
    --weights_dir    data/pi05/ckpts/3B \\
    --tokenizer_path data/paligemma/tokenizer.json \\
    --image_path     data/pi0/cam_high.jpg data/pi0/cam_left_wrist.jpg data/pi0/cam_right_wrist.jpg

Mirrors examples/pi05/demo.cpp: prefill() leaves the cameras and the prompt,
which carries the binned state, in the KV cache, and decode() denoises the
chunk from seeded noise.
"""

from __future__ import annotations

import os
import argparse

import pyvai
from pyvai.models import Pi05, Pi05Config


# The task of the C++ demo's observation (RoboTwin_clean, episode 0, frame 120).
PROMPT = "Take the shoe with patterned bottom from the table and place it on the mat"


def parse_args():
    parser = argparse.ArgumentParser(description="pi0.5 action chunk with pyvai (built-in models).")
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Folder holding the LeRobot pi0.5 *.safetensors shards (lerobot/pi05_base) "
        "(data/pi05/ckpts/<variant>).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to PaliGemma's tokenizer.json (pi0.5 reads its prompt with it).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        nargs="+",
        required=True,
        help="One image per camera, in the order the policy reads them (base, "
        "left wrist, right wrist for the 3B preset); fewer than it allows "
        "is a shorter prefix.",
    )
    parser.add_argument(
        "--preset", type=str, default="3B", choices=Pi05Config.presets, help="Checkpoint variant."
    )
    parser.add_argument("--prompt", type=str, default=PROMPT, help="The task string.")
    parser.add_argument(
        "--denoise_steps", type=int, default=10, help="Euler steps from noise to the action chunk."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the flow-matching noise the chunk starts from."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.weights_dir, args.tokenizer_path, *args.image_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    config = Pi05Config.preset(args.preset)
    if len(args.image_path) > config.max_cameras:
        raise SystemExit(
            f"--image_path: the {args.preset} policy takes at most "
            f"{config.max_cameras} cameras, got {len(args.image_path)}"
        )

    model = Pi05(config, w_default_dtype=pyvai.Dtype.F16, denoise_steps=args.denoise_steps)
    model.set_profile_tags()
    model.load_weights(args.weights_dir)
    model.load_tokenizer(args.tokenizer_path)

    for c, path in enumerate(args.image_path):
        print(f"Camera {c}: {path}")
    print(f'Task: "{args.prompt}"\n')

    # Proprioception, normalized to [-1, 1] and zero-padded to action_dim; the
    # prompt carries it binned. A policy on a robot feeds normalized joint
    # positions here; this is a stand-in for them.
    state = [0.1 * (i + 1) for i in range(6)]

    chunk = model.act(args.image_path, args.prompt, state, seed=args.seed)

    # The first steps of the chunk and its leading action dims, where an
    # embodiment's real degrees of freedom sit.
    print(f"=== Action chunk ({model.action_horizon} steps x {model.action_dim} dims) ===")
    for s in range(min(model.action_horizon, 5)):
        print(f"  t{s:<2} " + " ".join(f"{v: .4f}" for v in chunk[s, :8]) + " ...")
    print("  ...")

    model.print_profile()
    return 0


if __name__ == "__main__":
    main()
