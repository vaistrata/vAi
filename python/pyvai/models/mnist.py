"""MNIST — high-level wrapper over the C++-bound classifier.

The model is a copy of `examples/mnist/main.cpp` kept in the bindings, since that
file holds it together with main(); `predict(image)` is the run its main() does,
plus the argmax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._pyvai import MNISTNet, Tensor, Device, Dtype
from ._image import image_processor


__all__ = ["MNIST", "MNISTNet"]


class MNIST:
    """MNIST: bound C++ `MNISTNet` + [0, 1] normalize + argmax."""

    def __init__(
        self,
        w_default_dtype: Dtype = Dtype.F32,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
    ):
        self._weights_loaded = False
        self.net = MNISTNet(
            w_default_dtype=w_default_dtype, o_default_dtype=o_default_dtype, device=device
        )

    def load_weights(self, path: str | Path) -> "MNIST":
        """Load the checkpoint folder at `path` (every *.safetensors in it)."""
        self.net.load_weights(str(path))
        self._weights_loaded = True
        return self

    def logits(self, image) -> np.ndarray:
        """The 10 class logits of a 28x28 gray image: a file path, or a [H, W] uint8 array."""
        if not self._weights_loaded:
            raise RuntimeError("MNIST.logits: call load_weights(...) first")

        chw = image_processor(image, channels=1).normalize(0.0, 1.0).chw().to_numpy()
        if chw.shape != (1, 28, 28):
            raise ValueError(f"MNIST.logits: expected a 28x28 image, got {chw.shape[1:]}")
        return self.net(Tensor.from_numpy(chw)).to_numpy()

    def predict(self, image) -> int:
        """The digit in a 28x28 gray image, as logits() takes it."""
        return int(np.argmax(self.logits(image)))
