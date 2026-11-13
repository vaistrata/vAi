"""pi0 — high-level wrapper over the C++-bound vision-language-action model
+ tokenizer.

The model is `models/pi0/model.h` (C++); a chunk runs the way the C++ demo
(`examples/pi0/demo.cpp`) runs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .._pyvai import Pi0Model, Pi0Config, pi0_sample_noise, Tokenizer, Device, Dtype, Tensor, Align
from ._image import image_processor


__all__ = ["Pi0", "Pi0Model", "Pi0Config", "pi0_sample_noise"]


class Pi0:
    """pi0: bound C++ `Pi0Model` + `Tokenizer`, turning camera images, a task
    string and the robot's state into one action chunk. `config` picks the
    checkpoint (default: `Pi0Config.preset("3B")`; see `Pi0Config.presets`).
    act() runs a whole chunk; prefill() and decode() are its two halves, so
    several chunks can be decoded, from different seeds, over one observation."""

    def __init__(
        self,
        config: Pi0Config | None = None,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
        denoise_steps: int = 10,
        kv_cache_dtype: Dtype = Dtype.F16,
    ):
        self.config = config if config is not None else Pi0Config.preset("3B")
        self.model = Pi0Model(
            self.config,
            w_default_dtype=w_default_dtype,
            o_default_dtype=o_default_dtype,
            device=device,
            denoise_steps=denoise_steps,
            kv_cache_dtype=kv_cache_dtype,
        )
        self.tokenizer: Tokenizer | None = None
        self._weights_loaded = False

    def load_weights(self, path: str | Path) -> "Pi0":
        """Load the checkpoint folder at `path` (every *.safetensors in it)."""
        self.model.load_weights(str(path))
        self._weights_loaded = True
        return self

    def load_tokenizer(self, tokenizer_json: str | Path) -> "Pi0":
        """Load a `tokenizer.json`; pi0 reads its prompt with PaliGemma's."""
        self.tokenizer = Tokenizer(str(tokenizer_json))
        return self

    def set_profile_tags(self) -> "Pi0":
        """Turn on per-node GPU profiling."""
        self.model.set_profile_tags()
        return self

    def print_profile(self) -> None:
        self.model.print_profile()

    # -- geometry -------------------------------------------------------------------

    @property
    def max_cameras(self) -> int:
        return self.model.max_cameras

    @property
    def image_size(self) -> int:
        """Side of the square a camera is letterboxed into."""
        return self.model.image_size

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    @property
    def action_dim(self) -> int:
        return self.model.action_dim

    @property
    def denoise_steps(self) -> int:
        return self.model.denoise_steps

    # -- prefill and decode ---------------------------------------------------------------

    def preprocess(self, image) -> np.ndarray:
        """An image (a file path, or an HWC uint8 RGB array) letterboxed on
        black into the model's square and scaled to [-1, 1]: one [3, S, S]
        float32 camera."""
        return (
            image_processor(image)
            .letterbox(self.image_size, Align.CENTER, 0.0)
            .normalize(-1.0, 1.0)
            .chw()
            .to_numpy()
        )

    def prefill(self, images: Sequence, prompt: str, state) -> None:
        """Leave one observation in the KV cache. `images` holds 1 to
        max_cameras images (see preprocess()), in the order the policy reads
        them; `prompt` gets the trailing newline the checkpoint was trained
        with; `state` is the robot's proprioception, at most action_dim
        floats, zero-padded to it. An observation that would not leave the
        cache room for the chunk is refused before it runs."""
        tok = self._require_tokenizer("prefill")
        self._require_weights("prefill")
        if not 1 <= len(images) <= self.max_cameras:
            raise ValueError(
                f"Pi0.prefill: expected 1 to {self.max_cameras} camera images, got {len(images)}"
            )
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if vec.size > self.action_dim:
            raise ValueError(
                f"Pi0.prefill: the state has {vec.size} values, action_dim is {self.action_dim}"
            )
        padded = np.zeros((1, self.action_dim), dtype=np.float32)
        padded[0, : vec.size] = vec
        if not prompt.endswith("\n"):
            prompt += "\n"
        ids = tok.encode(prompt, add_special=True)  # BOS, as the C++ demo encodes
        cameras = np.stack([self.preprocess(image) for image in images])
        self.model.prefill(
            Tensor.from_numpy(cameras), Tensor.from_u32(ids), Tensor.from_numpy(padded)
        )

    def decode(self, seed: int = 0, noise: np.ndarray | None = None) -> np.ndarray:
        """Denoise one action chunk over the last prefill, in denoise_steps
        Euler steps. `noise` ([action_horizon, action_dim]) overrides the
        seeded start. Returns a float32 [action_horizon, action_dim] array."""
        self._require_weights("decode")
        shape = (self.action_horizon, self.action_dim)
        if noise is None:
            x = np.asarray(pi0_sample_noise(shape[0] * shape[1], seed), dtype=np.float32).reshape(
                shape
            )
        else:
            x = np.array(noise, dtype=np.float32).reshape(shape)
        return self.model.decode(Tensor.from_numpy(x)).to_numpy().reshape(shape)

    def act(self, images: Sequence, prompt: str, state, seed: int = 0) -> np.ndarray:
        """One action chunk for an observation: prefill() then decode(seed)."""
        self.prefill(images, prompt, state)
        return self.decode(seed)

    # -- guards ----------------------------------------------------------------------

    def _require_weights(self, who: str) -> None:
        if not self._weights_loaded:
            raise RuntimeError(f"Pi0.{who}: call load_weights(...) first")

    def _require_tokenizer(self, who: str) -> Tokenizer:
        if self.tokenizer is None:
            raise RuntimeError(f"Pi0.{who}: call load_tokenizer(...) first")
        return self.tokenizer
