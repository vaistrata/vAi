"""YOLOX — high-level wrapper over the C++-bound model + pre/post-processing.

The model is `models/yolox/` (C++) and the NMS postprocess the demo's
(`examples/yolox/processing.*`); `detect(image)` is the pipeline the C++ demo
(`examples/yolox/demo.cpp`) runs in main().
"""

from __future__ import annotations

from pathlib import Path

from .._pyvai import (
    YOLOXModel,
    YOLOXTier,
    YOLOXDetection,
    Tensor,
    Device,
    Dtype,
    Align,
    yolox_postprocess,
)
from ._image import image_processor


__all__ = ["YOLOX", "YOLOXModel", "YOLOXTier", "YOLOXDetection"]


class YOLOX:
    """YOLOX: bound C++ `YOLOXModel` + letterbox preprocess + NMS postprocess."""

    def __init__(
        self,
        tier: YOLOXTier = YOLOXTier.LARGE,
        image_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
    ):
        if o_default_dtype != Dtype.F32:
            raise ValueError(
                "YOLOX: detect() feeds float32 tensors, so o_default_dtype "
                "must be Dtype.F32 (use YOLOXModel directly for other hints)"
            )
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._weights_loaded = False
        self.model = YOLOXModel(
            tier,
            image_size,
            conf_threshold,
            iou_threshold,
            w_default_dtype=w_default_dtype,
            o_default_dtype=o_default_dtype,
            device=device,
        )

    def load_weights(self, path: str | Path) -> "YOLOX":
        """Load the checkpoint folder at `path` (every *.safetensors in it)."""
        self.model.load_weights(str(path))
        self._weights_loaded = True
        return self

    def set_profile_tags(self) -> "YOLOX":
        """Turn on per-node GPU profiling."""
        self.model.set_profile_tags()
        return self

    def detect(self, image) -> list[YOLOXDetection]:
        """Run detection on an image: a file path, or an HWC uint8 RGB array.
        Boxes come back in the image's own pixels."""
        if not self._weights_loaded:
            raise RuntimeError("YOLOX.detect: call load_weights(...) first")

        # BGR in 0..255, as the checkpoint expects.
        proc = image_processor(image).letterbox(self.image_size, Align.TOP_LEFT, 114.0).bgr().chw()
        out = self.model(Tensor.from_numpy(proc.to_numpy())).to_numpy().reshape(-1)

        # Objectness * class filter + NMS, mapped back to the image (C++).
        return list(
            yolox_postprocess(
                out, proc.geometry, self.conf_threshold, self.iou_threshold, self.model.num_classes
            )
        )

    def print_profile(self) -> None:
        self.model.print_profile()
