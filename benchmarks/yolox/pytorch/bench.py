#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WEIGHTS_DIR = (REPO_ROOT / "data/benchmarks/yolox").resolve()
DEFAULT_WARMUP = 1000
DEFAULT_ITERATIONS = 100
MODEL_SIZES = ("s", "m", "l", "x")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOX PyTorch benchmark.",
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", choices=MODEL_SIZES, default="l")
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data/images/dog.jpg",
        help="Image input. Loaded and preprocessed once before timing.",
    )
    parser.add_argument("--backend", default="cuda", choices=["cuda", "rocm", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Number of warmup iterations."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of benchmark iterations.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive.")
    return args


if __name__ == "__main__" and any(argument in ("-h", "--help") for argument in sys.argv[1:]):
    parse_args()
    raise SystemExit(0)


import numpy as np
import torch
import torch.nn as nn
from PIL import Image

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def class_name(self) -> str:
        return COCO_CLASSES[self.class_id]


class MegviiYOLOXBenchmarkModel(nn.Module):
    def __init__(self, weights_path: Path, image_size: int = 640):
        super().__init__()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Megvii checkpoint not found: {weights_path}")

        try:
            from yolox.exp import get_exp
        except ImportError as error:
            raise RuntimeError(
                "The PyTorch benchmark requires the official YOLOX package. "
                "Install the pinned source checkout documented in benchmarks/yolox/README.md."
            ) from error

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unexpected Megvii checkpoint format: {weights_path}")
        state_dict = checkpoint.get("model")
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Megvii checkpoint contains no model state_dict: {weights_path}")

        size = weights_path.parent.name
        if size not in MODEL_SIZES:
            raise RuntimeError(
                f"Cannot determine YOLOX size from model directory: {weights_path.parent}"
            )
        self.model = get_exp(exp_name=f"yolox-{size}").get_model().float().eval()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.head.decode_in_inference = False
        self.image_size = image_size
        if image_size % 32 != 0:
            raise ValueError("YOLOX benchmark image size must be divisible by 32")

        grids = []
        expanded_strides = []
        for stride in (8, 16, 32):
            size = image_size // stride
            y, x = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing="ij",
            )
            grids.append(torch.stack((x, y), dim=-1).reshape(1, -1, 2))
            expanded_strides.append(
                torch.full((1, size * size, 1), float(stride), dtype=torch.float32)
            )
        self.register_buffer("decode_grid", torch.cat(grids, dim=1), persistent=False)
        self.register_buffer("decode_stride", torch.cat(expanded_strides, dim=1), persistent=False)
        try:
            self.yolox_version = importlib.metadata.version("yolox")
        except importlib.metadata.PackageNotFoundError:
            self.yolox_version = "source checkout"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        prediction = output[0] if isinstance(output, tuple) else output
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"YOLOX forward returned {type(prediction).__name__}, expected Tensor")
        centers = (prediction[..., 0:2] + self.decode_grid) * self.decode_stride
        sizes = prediction[..., 2:4].exp() * self.decode_stride
        return torch.cat((centers, sizes, prediction[..., 4:]), dim=-1)


def load_rgb_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as error:
        raise RuntimeError(f"failed to load image with Pillow: {path}: {error}") from error


def preprocess_loaded_image(
    image: np.ndarray, target_size: int = 640
) -> tuple[np.ndarray, int, int, int, int, int, int]:
    src_h, src_w = image.shape[:2]
    scale = min(float(target_size) / float(src_w), float(target_size) / float(src_h))
    final_w = int(src_w * scale)
    final_h = int(src_h * scale)
    x_ratio = float(src_w) / float(final_w)
    y_ratio = float(src_h) / float(final_h)
    xs = np.maximum(0.0, (np.arange(final_w, dtype=np.float32) + 0.5) * x_ratio - 0.5)
    ys = np.maximum(0.0, (np.arange(final_h, dtype=np.float32) + 0.5) * y_ratio - 0.5)
    x1 = xs.astype(np.uint32)
    y1 = ys.astype(np.uint32)
    x2 = np.minimum(x1 + 1, src_w - 1)
    y2 = np.minimum(y1 + 1, src_h - 1)
    x_diff = (xs - x1.astype(np.float32))[None, :, None]
    y_diff = (ys - y1.astype(np.float32))[:, None, None]

    p11 = image[y1[:, None], x1[None, :]].astype(np.float32)
    p21 = image[y1[:, None], x2[None, :]].astype(np.float32)
    p12 = image[y2[:, None], x1[None, :]].astype(np.float32)
    p22 = image[y2[:, None], x2[None, :]].astype(np.float32)
    top = p11 * (1.0 - x_diff) + p21 * x_diff
    bottom = p12 * (1.0 - x_diff) + p22 * x_diff
    resized = (top * (1.0 - y_diff) + bottom * y_diff).astype(np.uint8)

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:final_h, :final_w, :] = resized[:, :, ::-1]
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    return chw, src_w, src_h, final_w, final_h, 0, 0


def preprocess_image(
    path: Path, target_size: int = 640
) -> tuple[np.ndarray, int, int, int, int, int, int]:
    return preprocess_loaded_image(load_rgb_image(path), target_size)


def iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def apply_nms(detections: list[Detection]) -> list[Detection]:
    detections.sort(key=lambda det: det.confidence, reverse=True)
    result = []
    suppressed = [False] * len(detections)
    for i, det in enumerate(detections):
        if suppressed[i]:
            continue
        result.append(det)
        for j in range(i + 1, len(detections)):
            if suppressed[j] or det.class_id != detections[j].class_id:
                continue
            if iou(det, detections[j]) > IOU_THRESHOLD:
                suppressed[j] = True
    return result


def detections_from_output(output: np.ndarray) -> list[Detection]:
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2 or output.shape[1] != 85:
        raise ValueError(f"expected output shape [N, 85], got {output.shape}")

    class_scores = output[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    confidence = output[:, 4] * class_scores[np.arange(output.shape[0]), class_ids]
    detections = []
    for i in range(output.shape[0]):
        if confidence[i] < CONF_THRESHOLD:
            continue

        detections.append(
            Detection(
                float(output[i, 0] - output[i, 2] * 0.5),
                float(output[i, 1] - output[i, 3] * 0.5),
                float(output[i, 0] + output[i, 2] * 0.5),
                float(output[i, 1] + output[i, 3] * 0.5),
                float(confidence[i]),
                int(class_ids[i]),
            )
        )
    return apply_nms(detections)


def tensor_to_numpy_output(output: torch.Tensor) -> np.ndarray:
    return output[0].detach().float().cpu().numpy().astype(np.float32, copy=True)


def print_detections(detections: list[Detection]) -> None:
    print(f"\n=== Detections ({len(detections)}) ===")
    for index, det in enumerate(detections):
        print(
            f"[{index}] {det.class_name} ({det.confidence * 100.0:.2f}%): "
            f"[{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}]"
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_backend(backend: str, gpu_index: int) -> tuple[torch.device, str, str | None]:
    if backend == "cpu":
        return torch.device("cpu"), "CPU", None

    if gpu_index < 0:
        raise ValueError("--gpu must be non-negative")

    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    if backend == "rocm":
        if not hip_version:
            raise RuntimeError(
                "ROCm requested but this PyTorch build does not report torch.version.hip"
            )
        backend_label = "ROCm"
    else:
        if hip_version:
            raise RuntimeError(
                "CUDA requested, but this PyTorch build is ROCm/HIP. "
                "Use a CUDA PyTorch environment for NVIDIA GPUs."
            )
        if not cuda_version:
            raise RuntimeError(
                "CUDA requested but this PyTorch build does not report torch.version.cuda"
            )
        backend_label = "CUDA"

    if not torch.cuda.is_available():
        raise RuntimeError(f"{backend.upper()} requested but torch.cuda.is_available() is false")
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU index {gpu_index} requested, but only {torch.cuda.device_count()} CUDA/HIP device(s) are available"
        )

    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    return device, backend_label, hip_version


def size_weight_path(size: str) -> Path:
    return DEFAULT_WEIGHTS_DIR / f"yolox_{size}.pth"


def main() -> int:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    device, backend_label, hip_version = resolve_backend(args.backend, args.gpu)
    if args.precision == "fp16" and device.type != "cuda":
        raise RuntimeError("--precision fp16 is supported only for CUDA/ROCm PyTorch backends.")
    tensor_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    weights_path = size_weight_path(args.size)

    print("========================================")
    print("  YOLOX PyTorch Benchmark")
    print("========================================")
    print(f"Size          : YOLOX-{args.size}")
    print(f"Warmup        : {args.warmup}, Iterations: {args.iterations}")
    print(f"Device        : {device}")
    print(f"Backend       : {backend_label}")
    print(f"Precision     : {args.precision}")
    if device.type == "cuda":
        print(f"GPU           : {args.gpu} ({torch.cuda.get_device_name(device)})")
    if hip_version:
        print(f"HIP Version   : {hip_version}")
    print(
        "Measurement   : official YOLOX forward plus equivalent functional decode; "
        "no H2D/D2H, confidence filtering, or NMS"
    )
    print(f"Image Path    : {args.image}")
    print(f"Weights Path  : {weights_path}")

    model = MegviiYOLOXBenchmarkModel(weights_path).eval()
    print(f"YOLOX         : {model.yolox_version}")
    model.to(device=device, dtype=tensor_dtype)

    chw, src_w, src_h, *_ = preprocess_image(args.image)
    print(f"\n[Input] Preprocessed image before benchmark: {src_w}x{src_h} -> 3x640x640 (CHW)")
    input_cpu = torch.from_numpy(chw).unsqueeze(0).contiguous()
    input_device = input_cpu.to(device=device, dtype=tensor_dtype)

    bench_name = f"PyTorch_{backend_label}_{args.precision.upper()}_YOLOX_{args.size}_inference"

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_device)
        synchronize(device)
        elapsed = []
        print(f"[{bench_name}] Running {args.iterations} iterations...")
        if device.type == "cuda":
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(input_device)
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
        else:
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                output = model(input_device)
                synchronize(device)
                elapsed.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(elapsed, dtype=np.float64)
    print("-----------------------------------")
    print(f"[{bench_name}] Result")
    print(f" - Processed Items : {len(times)}")
    print(f" - Avg Latency     : {times.mean():.6f} ms")
    print(f" - Min Latency     : {times.min():.6f} ms")
    print(f" - Max Latency     : {times.max():.6f} ms")
    print(f" - Std Latency     : {times.std():.6f} ms")
    print(f" - Throughput      : {1000.0 / times.mean():.6f} items/sec (FPS or TPS)")
    print("-----------------------------------")

    synchronize(device)
    print_detections(detections_from_output(tensor_to_numpy_output(output)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
