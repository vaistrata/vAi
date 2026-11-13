#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PYTORCH_BENCH_DIR = REPO_ROOT / "benchmarks" / "yolox" / "pytorch"
if str(PYTORCH_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BENCH_DIR))

from bench import MegviiYOLOXBenchmarkModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the official YOLOX model to ncnn with folded Focus and pnnx."
    )
    parser.add_argument("--sizes", nargs="+", choices=("s", "m", "l", "x"), default=["l"])
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--weights-dir", type=Path, default=(REPO_ROOT / "data/benchmarks/yolox").resolve()
    )
    parser.add_argument(
        "--output-dir", type=Path, default=(REPO_ROOT / "data/benchmarks/yolox").resolve()
    )
    return parser.parse_args()


def _fold_focus(focus: torch.nn.Module) -> torch.nn.Module:
    from yolox.models.network_blocks import Focus

    if not isinstance(focus, Focus):
        raise ValueError("Expected the official YOLOX Focus stem for ncnn export.")
    # Official YOLOX S/M/L/X uses a 12-channel 3x3 stem with stride 1 and padding 1.
    conv = focus.conv.conv

    folded = torch.nn.Conv2d(
        3,
        conv.out_channels,
        kernel_size=6,
        stride=2,
        padding=2,
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    ).train(conv.training)
    with torch.no_grad():
        # Focus concatenates even/even, odd/even, even/odd, odd/odd pixels.
        # Interleave those input-channel groups into one stride-2 kernel.
        for part, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            folded.weight[:, :, dy::2, dx::2].copy_(conv.weight[:, part * 3 : (part + 1) * 3])
        if conv.bias is not None:
            folded.bias.copy_(conv.bias)

    # Retain the trained BatchNorm and activation; only replace Focus + Conv.
    focus.conv.conv = folded
    return focus.conv


def export_one(
    size: str,
    image_size: int,
    weights_dir: Path,
    output_dir: Path,
    pnnx: str,
) -> None:
    size_output_dir = output_dir
    weights_path = weights_dir / f"yolox_{size}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    model = MegviiYOLOXBenchmarkModel(weights_path, image_size=image_size).eval().cpu()
    backbone = model.model.backbone.backbone
    backbone.stem = _fold_focus(backbone.stem)
    example = torch.ones((1, 3, image_size, image_size), dtype=torch.float32)

    size_output_dir.mkdir(parents=True, exist_ok=True)
    final_param = size_output_dir / f"yolox_{size}.param"
    final_bin = size_output_dir / f"yolox_{size}.bin"
    with tempfile.TemporaryDirectory(prefix=f"yolox-{size}-ncnn-", dir=size_output_dir) as raw_dir:
        staging_dir = Path(raw_dir)
        onnx_path = staging_dir / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                onnx_path,
                input_names=["images"],
                output_names=["output0"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        staged_param = staging_dir / final_param.name
        staged_bin = staging_dir / final_bin.name
        result = subprocess.run(
            [
                pnnx,
                str(onnx_path),
                f"inputshape=[1,3,{image_size},{image_size}]f32",
                "fp16=0",
                f"pnnxparam={staging_dir / 'model.pnnx.param'}",
                f"pnnxbin={staging_dir / 'model.pnnx.bin'}",
                f"pnnxpy={staging_dir / 'model_pnnx.py'}",
                f"pnnxonnx={staging_dir / 'model.pnnx.onnx'}",
                f"ncnnparam={staged_param}",
                f"ncnnbin={staged_bin}",
                f"ncnnpy={staging_dir / 'model_ncnn.py'}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout, end="", flush=True)
        result.check_returncode()
        # pnnx can exit successfully after reporting unsupported operations.
        diagnostics = result.stdout.lower()
        if "not supported" in diagnostics or "unsupported" in diagnostics:
            raise RuntimeError(
                "pnnx reported an unsupported operation in the official YOLOX model; "
                "ncnn files were not published."
            )
        os.replace(staged_param, final_param)
        os.replace(staged_bin, final_bin)
    print(f"[done] wrote {final_param} and {final_bin}")


def main() -> int:
    args = parse_args()
    pnnx = shutil.which("pnnx")
    if pnnx is None:
        raise FileNotFoundError("pnnx was not found in PATH")
    for size in args.sizes:
        export_one(
            size,
            args.image_size,
            args.weights_dir.resolve(),
            args.output_dir.resolve(),
            pnnx,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
