"""MNIST built by hand with the node DSL.

As the C++ models do: subclass `pyvai.NeuralNet`, create nodes from
`pyvai.nodes` and wire them with `-`. The weight keys are this graph's own;
the demo uses the committed MNIST checkpoint's `conv1 / conv2 / fc`:

    python python/examples/mnist/main.py \\
        --weights_path data/ckpts/mnist/jroot/model.safetensors \\
        --image_path   data/images/0.png
"""

from __future__ import annotations

import os
import argparse

import numpy as np

import pyvai
from pyvai import nodes


# fp16 weights, fp32 I/O (the C++ demo runs fp32 weights: use Dtype.F32 to
# reproduce its logits exactly).
WEIGHT_DTYPE = pyvai.Dtype.F16
INOUT_DTYPE = pyvai.Dtype.F32


class MnistNet(pyvai.NeuralNet):
    """Hand-built MNIST convnet using the node DSL (mirrors examples/mnist/main.cpp)."""

    def __init__(self):
        super().__init__(w_default_dtype=WEIGHT_DTYPE)
        self.conv0 = nodes.ConvolutionNode(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, epilogue=pyvai.Epilogue.RELU
        )
        self.pool0 = nodes.MaxPool2dNode(kernel_size=2, stride=2)
        self.conv1 = nodes.ConvolutionNode(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, epilogue=pyvai.Epilogue.RELU
        )
        self.pool1 = nodes.MaxPool2dNode(kernel_size=2, stride=2)
        self.reshape = nodes.ReshapeNode([7 * 7 * 64])
        self.linear = nodes.LinearNode(in_dim=7 * 7 * 64, out_dim=10)

        self._build_graph()

    def _build_graph(self) -> None:
        """`-` connects the left node's output slot to the right one's input
        slot and yields the right node, so the chain keeps going."""
        (
            self.input("in0")
            - self.conv0
            - self.pool0
            - self.conv1
            - self.pool1
            - self.reshape
            - self.linear
            - self.output("out0")
        )


def load_weights(net: MnistNet, path: str) -> None:
    """Checkpoint keys -> node slots; pyvai.Weights reads the format off the suffix."""
    w = pyvai.Weights(path)
    net.conv0["weight"] = w.get("conv1.weight", dtype=WEIGHT_DTYPE)
    net.conv0["bias"] = w.get("conv1.bias", dtype=WEIGHT_DTYPE)
    net.conv1["weight"] = w.get("conv2.weight", dtype=WEIGHT_DTYPE)
    net.conv1["bias"] = w.get("conv2.bias", dtype=WEIGHT_DTYPE)
    net.linear["weight"] = w.get("fc.weight", dtype=WEIGHT_DTYPE)
    net.linear["bias"] = w.get("fc.bias", dtype=WEIGHT_DTYPE)


def load_image_chw(path: str) -> np.ndarray:
    """The 28x28 gray image, normalized to [0, 1], as [1, 28, 28] — C, H, W."""
    return pyvai.ImageProcessor(path, channels=1).normalize(0.0, 1.0).chw().to_numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST via the low-level vAi node DSL.")
    parser.add_argument(
        "--weights_path", type=str, required=True, help="MNIST safetensors (keys: conv1/conv2/fc)."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to a 28x28 grayscale input image."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"weights not found: {args.weights_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"image not found: {args.image_path}")

    net = MnistNet()
    # As the C++ demo's loadWeights<"jroot">: a loader mapping file keys to slots.
    load_weights(net, args.weights_path)

    img = load_image_chw(args.image_path)
    logits = net(pyvai.Tensor.from_numpy(img, dtype=INOUT_DTYPE)).to_numpy()

    print(f"input  : {img.shape} {img.dtype}")
    for i, v in enumerate(logits):
        print(f"  data[{i}] = {v:.6f}")
    print(f"argmax : {int(np.argmax(logits))}")
    return 0


if __name__ == "__main__":
    main()
