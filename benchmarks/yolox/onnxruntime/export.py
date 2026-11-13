#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PYTORCH_BENCH_DIR = REPO_ROOT / "benchmarks" / "yolox" / "pytorch"
if str(PYTORCH_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BENCH_DIR))

from bench import MegviiYOLOXBenchmarkModel

DEFAULT_WEIGHTS_DIR = (REPO_ROOT / "data/benchmarks/yolox").resolve()
DEFAULT_SIZES = ("s", "m", "l", "x")
DEFAULT_OPSET = 17
FP16_NO_BLOCKED_OPS = ["__vai_no_blocked_operators__"]


def validate_model_precision(model: onnx.ModelProto, precision: str) -> None:
    expected_type = onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
    for label, values in (("input", model.graph.input), ("output", model.graph.output)):
        if len(values) != 1:
            raise RuntimeError(f"expected exactly one ONNX graph {label}, found {len(values)}")
        element_type = values[0].type.tensor_type.elem_type
        if element_type != expected_type:
            raise RuntimeError(
                f"ONNX {label} dtype mismatch for {precision}: "
                f"expected {expected_type}, got {element_type}"
            )


def make_temporary_onnx_path(output_dir: Path, output_path: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.onnx",
        dir=output_dir,
    )
    os.close(descriptor)
    return Path(raw_path)


def convert_fp32_model_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
    except ImportError as error:
        raise RuntimeError("FP16 ONNX export requires the Python onnxruntime package.") from error

    converted_model = convert_float_to_float16(
        model,
        keep_io_types=False,
        # The converter replaces an empty list with its default block list.
        # A non-matching sentinel keeps every YOLOX operator in FP16.
        op_block_list=FP16_NO_BLOCKED_OPS,
    )
    # ONNX Runtime 1.24 appends the mandatory Resize scale-input Cast nodes.
    # Restore graph order before running the ONNX checker or publishing.
    OnnxModel(converted_model).topological_sort(is_deterministic=True)
    return converted_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOX raw-output benchmark models to ONNX."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=DEFAULT_SIZES,
        default=list(DEFAULT_SIZES),
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    return parser.parse_args()


def export_one(
    size: str,
    weights_dir: Path,
    output_dir: Path,
    image_size: int,
    opset: int,
    precision: str,
) -> Path:
    weights_path = weights_dir / f"yolox_{size}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    output_path = output_dir / f"yolox_{size}.{precision}.onnx"
    model = (
        MegviiYOLOXBenchmarkModel(
            weights_path,
            image_size=image_size,
        )
        .eval()
        .cpu()
    )
    example = torch.ones(
        (1, 3, image_size, image_size),
        dtype=torch.float32,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] yolox_{size} precision={precision}: {weights_path} -> {output_path}")
    temporary_paths: list[Path] = []
    try:
        fp32_path = make_temporary_onnx_path(output_dir, output_path)
        temporary_paths.append(fp32_path)
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                fp32_path,
                input_names=["images"],
                output_names=["output0"],
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )

        exported_model = onnx.load(fp32_path)
        if precision == "fp16":
            exported_model = convert_fp32_model_to_fp16(exported_model)

        validate_model_precision(exported_model, precision)
        onnx.checker.check_model(exported_model)

        publish_path = make_temporary_onnx_path(output_dir, output_path)
        temporary_paths.append(publish_path)
        onnx.save(exported_model, publish_path)
        os.replace(publish_path, output_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    print(f"[done] wrote {output_path}")
    return output_path


def main() -> int:
    args = parse_args()

    weights_dir = DEFAULT_WEIGHTS_DIR
    output_dir = DEFAULT_WEIGHTS_DIR

    failures: list[tuple[str, str]] = []
    for size in args.sizes:
        try:
            export_one(
                size=size,
                weights_dir=weights_dir,
                output_dir=output_dir,
                image_size=args.image_size,
                opset=DEFAULT_OPSET,
                precision=args.precision,
            )
        except Exception as error:
            failures.append((size, str(error)))
            print(f"[fail] yolox_{size}: {error}")

    if failures:
        print("\nFailures")
        for size, message in failures:
            print(f"  yolox_{size}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
