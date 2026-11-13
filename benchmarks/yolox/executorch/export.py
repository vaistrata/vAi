import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.utils import DEFAULT_TEXTURE_LIMITS

try:
    from executorch.backends.vulkan.utils import SMALL_TEXTURE_LIMITS
except ImportError:
    # ExecuTorch 1.4 accepts ``small_texture_limits`` but does not export the
    # corresponding tuple from utils.py. Its Vulkan preprocessor uses 2048 in
    # every dimension when that option is enabled.
    SMALL_TEXTURE_LIMITS = (2048, 2048, 2048)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PYTORCH_BENCH_DIR = REPO_ROOT / "benchmarks" / "yolox" / "pytorch"
if str(PYTORCH_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_BENCH_DIR))

from bench import MegviiYOLOXBenchmarkModel

DEFAULT_SIZES = ("s", "m", "l", "x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOX s/m/l/x models to ExecuTorch .pte files."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        choices=list(DEFAULT_SIZES),
        help="YOLOX sizes to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "data/benchmarks/yolox").resolve(),
        help="Directory where .pte files will be written.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Optional data root containing yolox_<size>.pth files.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Model input height used during export.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Model input width used during export.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Model/input precision to export.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Target Vulkan GPU index from vulkaninfo --summary (default: 0).",
    )
    return parser.parse_args()


def get_vulkan_compile_options(gpu: int) -> dict[str, bool]:
    if gpu < 0:
        raise RuntimeError("--gpu must be non-negative.")

    with tempfile.TemporaryDirectory(prefix="yolox-vulkan-") as directory:
        output_path = Path(directory) / "device.json"
        try:
            subprocess.run(
                ["vulkaninfo", f"--json={gpu}", "--output", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "vulkaninfo was not found. Install Vulkan Tools/SDK and add its bin "
                "directory to PATH to query the target GPU."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("vulkaninfo timed out while querying the target GPU.") from error
        except subprocess.CalledProcessError as error:
            details = "\n".join(value.strip() for value in (error.stdout, error.stderr) if value)
            raise RuntimeError(f"Failed to query Vulkan GPU {gpu}:\n{details}") from error

        try:
            profile = json.loads(output_path.read_text(encoding="utf-8"))
            properties = profile["capabilities"]["device"]["properties"][
                "VkPhysicalDeviceProperties"
            ]
            device_name = properties["deviceName"]
            texture_limit = properties["limits"]["maxImageDimension3D"]
            if not isinstance(device_name, str) or not device_name:
                raise ValueError("deviceName is missing or invalid")
            if type(texture_limit) is not int or texture_limit <= 0:
                raise ValueError("maxImageDimension3D must be a positive integer")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"Cannot read GPU {gpu} maxImageDimension3D from vulkaninfo: {error}"
            ) from error

    if texture_limit < max(SMALL_TEXTURE_LIMITS):
        raise RuntimeError(
            f"GPU {gpu} ({device_name}) maxImageDimension3D={texture_limit} is below "
            f"ExecuTorch's small texture limits {SMALL_TEXTURE_LIMITS}."
        )

    small_texture_limits = texture_limit < max(DEFAULT_TEXTURE_LIMITS)
    print(
        f"[vulkan] GPU {gpu}: {device_name}; maxImageDimension3D={texture_limit}; "
        f"small_texture_limits={small_texture_limits}"
    )
    return {"small_texture_limits": small_texture_limits}


def resolve_weights_path(
    size: str,
    weights_dir: Path | None,
    output_dir: Path,
) -> Path:
    relative_path = Path(f"yolox_{size}.pth")
    candidates: list[Path] = []
    if weights_dir is not None:
        candidates.append(weights_dir / relative_path)
    candidates.append(output_dir / relative_path)
    candidates.append((REPO_ROOT / "data/benchmarks/yolox" / relative_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"YOLOX weights not found: {relative_path}; searched: {searched}")


def make_example_inputs(
    height: int,
    width: int,
    precision: str,
) -> tuple[torch.Tensor]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    return (torch.ones((1, 3, height, width), dtype=dtype),)


def output_path_for(
    output_dir: Path,
    size: str,
    precision: str,
) -> Path:
    return output_dir / f"yolox_{size}.{precision}.pte"


def write_program(executorch_program, output_path: Path) -> None:
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=f".{output_path.stem}.",
    ) as staging_dir_name:
        staging_dir = Path(staging_dir_name)
        staged_program = staging_dir / output_path.name
        with open(staged_program, "wb") as file:
            executorch_program.write_to_file(file)
        os.replace(staged_program, output_path)


def export_one(
    size: str,
    output_dir: Path,
    weights_dir: Path | None,
    height: int,
    width: int,
    precision: str,
    compile_options: dict[str, bool],
) -> Path:
    if height != width:
        raise ValueError("YOLOX ExecuTorch export currently requires square input dimensions.")

    output_path = output_path_for(output_dir, size, precision)
    weights_path = resolve_weights_path(size, weights_dir, output_dir)
    print(f"[load] size={size} backend=vulkan precision={precision} source={weights_path}")
    pt_model = MegviiYOLOXBenchmarkModel(
        weights_path,
        image_size=height,
    ).eval()
    if precision == "fp16":
        pt_model.half()
    example_args = make_example_inputs(height, width, precision)

    with torch.no_grad():
        aten_dialect = torch.export.export(pt_model, args=example_args, strict=True)

    edge_program = to_edge_transform_and_lower(
        aten_dialect,
        partitioner=[VulkanPartitioner(compile_options=compile_options)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=False,
        ),
    )
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    write_program(executorch_program, output_path)

    print(f"[done] wrote {output_path}")
    del executorch_program
    del edge_program
    del aten_dialect
    del pt_model
    gc.collect()
    return output_path


def main() -> int:
    args = parse_args()
    try:
        compile_options = get_vulkan_compile_options(args.gpu)
    except RuntimeError as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1

    successes: list[Path] = []
    failures: list[tuple[str, str]] = []

    for size in args.sizes:
        try:
            result = export_one(
                size=size,
                output_dir=args.output_dir,
                weights_dir=args.weights_dir,
                height=args.height,
                width=args.width,
                precision=args.precision,
                compile_options=compile_options,
            )
            successes.append(result)
        except Exception as error:
            failures.append((size, str(error)))
            print(f"[fail] size={size}: {error}")
            traceback.print_exc()

    print("\nSummary")
    print(f"  success: {len(successes)}")
    for path in successes:
        print(f"    {path}")

    print(f"  failed: {len(failures)}")
    for size, message in failures:
        print(f"    yolox_{size}: {message}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
