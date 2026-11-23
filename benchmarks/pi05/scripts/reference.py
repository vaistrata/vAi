#!/usr/bin/env python3
"""LeRobot pi0.5 FP32 reference for the vAi bench's observation (benchmarks/common/bench-inputs.h).

The observation - the first --cameras cameras, the task, the state - goes through LeRobot's own
preprocessing (the checkpoint's policy_preprocessor.json, then PI05Policy._preprocess_images) and
the model's own seeded noise. Written as little-endian FP32, each named <kind>_<cameras>_<chunk>_<steps>.bin:

    reference   the model's normalized [chunk, 32] actions
    noise       the [chunk, 32] noise they start from
    images      the [cameras, 3, 224, 224] pixels the model saw, CHW in [-1, 1]

The benches take all three, so they compare the model alone, not the image preprocessing.

--policy openpi swaps in the prompt of openpi's pi05_libero, which TensorRT Edge-LLM serves: the
task alone, without the state. The bench configurations (common/vla-config.h) name the values to run:

    MIGraphX, vla.cpp   --cameras 2 --chunk 50 --steps 10
    TensorRT            --cameras 2 --chunk 10 --steps 10 --policy openpi
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

DATA = Path(__file__).resolve().parents[3] / "data"
CKPT = DATA / "ckpts/pi05/lerobot/3B"  # the benches' checkpoint: lerobot/pi05_libero_base
OUT_DIR = DATA / "benchmarks/pi05/3B"

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
TASK = "Take the shoe with patterned bottom from the table and place it on the mat"
STATE = [0.1 * (i + 1) for i in range(6)] + [0.0] * 26
SEED = 0


def load_camera(name):  # as a LeRobotDataset frame holds it: CHW in [0, 1]
    pixels = np.asarray(Image.open(DATA / f"images/{name}.jpg").convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255


def openpi_prompt(preprocess, length, device):
    """openpi's PaligemmaTokenizer without a state: the cleaned task, then "\\n" on its own."""
    tokenizer = next(
        step.input_tokenizer for step in preprocess.steps if hasattr(step, "input_tokenizer")
    )
    task = TASK.strip().replace("_", " ").replace("\n", " ")
    ids = tokenizer.encode(task, add_special_tokens=True) + tokenizer.encode(
        "\n", add_special_tokens=False
    )
    tokens = torch.zeros((1, length), dtype=torch.long, device=device)
    masks = torch.zeros((1, length), dtype=torch.bool, device=device)
    tokens[0, : len(ids)] = torch.tensor(ids, device=device)
    masks[0, : len(ids)] = True
    return tokens, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cameras",
        type=int,
        default=3,
        choices=range(1, len(CAMERAS) + 1),
        help="images, the first of %(choices)s; the rest are empty cameras",
    )
    parser.add_argument("--chunk", type=int, default=50, help="actions per chunk")
    parser.add_argument("--steps", type=int, default=10, help="denoising steps")
    parser.add_argument(
        "--policy",
        choices=["lerobot", "openpi"],
        default="lerobot",
        help="the prompt: lerobot's carries the discretized state, openpi's pi05_libero the task alone",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PI05Policy.from_pretrained(CKPT).float().to(device).eval()
    config = policy.config  # the model reads both from here
    config.chunk_size, config.num_inference_steps = args.chunk, args.steps

    preprocess, _ = make_pre_post_processors(
        config, CKPT, preprocessor_overrides={"device_processor": {"device": str(device)}}
    )

    # The checkpoint's camera keys, in order, take the bench's cameras.
    cameras = zip(config.image_features, CAMERAS[: args.cameras])
    batch = preprocess(
        {key: load_camera(name) for key, name in cameras}
        | {OBS_STATE: torch.tensor(STATE), "task": TASK}
    )
    tokens, token_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    if args.policy == "openpi":
        tokens, token_masks = openpi_prompt(preprocess, tokens.shape[1], tokens.device)

    # What sample_actions draws without a noise argument; kept to write it out.
    torch.manual_seed(SEED)
    noise = policy.model.sample_noise((1, config.chunk_size, config.max_action_dim), device)

    # PI05Policy.predict_action_chunk, without its cut to the dataset's action width.
    with torch.inference_mode():
        images, image_masks = policy._preprocess_images(batch)
        actions = policy.model.sample_actions(images, image_masks, tokens, token_masks, noise=noise)

    # The given cameras' pixels, without the checkpoint's empty slots, which their masks keep out.
    pixels = torch.cat([image for image, mask in zip(images, image_masks) if mask.item()])
    assert pixels.shape == (args.cameras, 3, *config.image_resolution), pixels.shape

    suffix = f"{args.cameras}_{args.chunk}_{args.steps}.bin"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = OUT_DIR / f"reference_{suffix}"

    actions.float().cpu().numpy().astype("<f4").tofile(reference)
    # The benches start from this noise: sampleNoise does not reproduce torch's RNG.
    noise.float().cpu().numpy().astype("<f4").tofile(OUT_DIR / f"noise_{suffix}")
    # And from these pixels: LeRobot's resize_with_pad is not vAi's letterbox.
    pixels.float().cpu().numpy().astype("<f4").tofile(OUT_DIR / f"images_{suffix}")

    print(
        f"{reference}: {tuple(actions.shape)}, "
        f"{sum(m.item() for m in image_masks)} cameras, "
        f"{int(token_masks.sum())} of {tokens.shape[1]} prompt tokens"
    )


if __name__ == "__main__":
    main()
