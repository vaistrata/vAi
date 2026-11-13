import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


CKPT_DIR = Path(__file__).resolve().parents[3] / "data" / "ckpts" / "yolox" / "megvii"
TIERS = ["small", "medium", "large", "xlarge"]


def convert(checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"{checkpoint_path} does not contain a YOLOX 'model' state dict")

    state = {}
    for key, tensor in checkpoint["model"].items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{checkpoint_path}: model[{key!r}] is not a tensor")
        state[key] = (
            tensor.float().contiguous() if tensor.is_floating_point() else tensor.contiguous()
        )

    output_path = checkpoint_path.with_suffix(".safetensors")
    save_file(state, output_path)
    print(
        f"{checkpoint_path.parent.name}/{checkpoint_path.name} -> {output_path.name} ({len(state)} tensors)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data/ckpts/yolox/megvii/<tier>/model.pth to model.safetensors in place."
    )
    parser.add_argument(
        "tiers",
        nargs="*",
        choices=TIERS,
        metavar="tier",
        help=f"tiers to convert ({'|'.join(TIERS)}); default: every tier whose model.pth is present",
    )
    args = parser.parse_args()

    tiers = args.tiers or [tier for tier in TIERS if (CKPT_DIR / tier / "model.pth").is_file()]
    if not tiers:
        parser.error(f"no <tier>/model.pth found under {CKPT_DIR}")

    for tier in tiers:
        path = CKPT_DIR / tier / "model.pth"
        if not path.is_file():
            parser.error(f"checkpoint not found: {path}")
        convert(path)


if __name__ == "__main__":
    main()
