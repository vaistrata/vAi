"""Gemma 3 — high-level wrapper over the C++-bound model + tokenizer.

The model is `models/production/gemma3/net.h` (C++); generate() / stream() /
iter_tokens() come from `_CausalLM`, and chat() opens a session on the Gemma
instruct template. The 4B and 12B checkpoints are VLMs: an image becomes
`tokens_per_image` soft tokens of the prompt, placed as in
`models/production/gemma3/demo.cpp` (see iter_tokens()) or `chat.cpp` (see chat()).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator

import numpy as np

from .._pyvai import (
    Gemma3Model,
    Gemma3Config,
    Gemma3VisionEncoderConfig,
    Device,
    Dtype,
    Resize,
    Tensor,
    gemma3_expand_image_placeholders,
    gemma3_take_image_tags,
    gemma3_load_images,
)
from ._causal_lm import _CausalLM, _decode_stream
from ._image import image_processor
from .chat import ChatSession, ChatTemplate


__all__ = [
    "Gemma3",
    "Gemma3Model",
    "Gemma3Config",
    "Gemma3VisionEncoderConfig",
    "GEMMA3_CHAT_TEMPLATE",
    "Gemma3ChatSession",
    "gemma3_expand_image_placeholders",
    "gemma3_take_image_tags",
    "gemma3_load_images",
]


# The Gemma instruct format, as tokenizer_config.json's chat_template renders
# it: the assistant is "model", and the newline after a closed turn belongs to
# the template. generation_config.json names the two end-of-generation ids.
GEMMA3_CHAT_TEMPLATE = ChatTemplate(
    bos="<bos>",
    user_header="<start_of_turn>user\n",
    assistant_header="<start_of_turn>model\n",
    end_of_turn="<end_of_turn>\n",
    eog_pieces=("<eos>", "<end_of_turn>"),
)


class Gemma3(_CausalLM):
    """Gemma 3: bound C++ `Gemma3Model` + `Tokenizer` with the completion API
    and chat(). `config` picks the checkpoint (default: `Gemma3Config.preset("4B")`;
    see `Gemma3Config.presets`); `"4B"` and `"12B"` take images as well."""

    _bos_piece = "<bos>"
    _eos_pieces = ("<eos>", "<end_of_turn>")

    def __init__(
        self,
        config: Gemma3Config | None = None,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
        max_runtime_context: int | None = 4096,
        kv_cache_dtype: Dtype = Dtype.F16,
    ):
        self.config = config if config is not None else Gemma3Config.preset("4B")
        super().__init__(
            Gemma3Model(
                self.config,
                w_default_dtype=w_default_dtype,
                o_default_dtype=o_default_dtype,
                device=device,
                max_runtime_context=max_runtime_context,
                kv_cache_dtype=kv_cache_dtype,
            )
        )

    # -- images (VLM checkpoints) -------------------------------------------------

    @property
    def has_vision(self) -> bool:
        """Whether the checkpoint has a vision encoder (config.vision)."""
        return self.config.vision is not None

    def preprocess(self, images) -> Tensor:
        """One image or a sequence of them (file paths or HWC uint8 RGB
        arrays), squashed to the vision encoder's square as Gemma3ImageProcessor
        does and scaled to [-1, 1]: the [B, 3, S, S] float Tensor prefill
        takes. A Tensor passes through."""
        if isinstance(images, Tensor):
            return images
        if not self.has_vision:
            raise ValueError(
                f"Gemma3.preprocess: the {self.config.variant} checkpoint has no vision encoder"
            )
        if isinstance(images, (str, os.PathLike, np.ndarray)):
            images = [images]
        if not images:
            raise ValueError("Gemma3.preprocess: no images")
        size = self.config.vision.image_size
        return Tensor.from_numpy(
            np.stack(
                [
                    image_processor(image)
                    .resize(size, Resize.BILINEAR_ANTIALIAS_INT16)
                    .normalize(-1.0, 1.0)
                    .chw()
                    .to_numpy()
                    for image in images
                ]
            )
        )

    def _encode(self, text: str, add_bos: bool) -> list[int]:
        # A VLM prompt names an image with <start_of_image>; the model wants
        # the whole placeholder block the processor writes for it.
        if self.has_vision:
            text = gemma3_expand_image_placeholders(text, self.config.vision.tokens_per_image)
        return super()._encode(text, add_bos)

    def _prefill(self, ids: list[int], image: Tensor | None) -> Tensor:
        return self.model.prefill(image, Tensor.from_u32(ids))

    # -- completion, with optional images ------------------------------------------

    def iter_tokens(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        images=None,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[int]:
        """As _CausalLM.iter_tokens. On a VLM checkpoint every
        <start_of_image> in `prompt` stands for one image, and `images` (see
        preprocess()) brings them in that order; the model raises
        RuntimeError when the two do not match."""
        if images is None:
            yield from super().iter_tokens(
                prompt, max_new_tokens, stop_ids, reset_cache=reset_cache, add_bos=add_bos
            )
            return
        self._require_ready("iter_tokens")
        if add_bos is None:
            add_bos = reset_cache
        yield from self._iter_ids(
            self._encode(prompt, add_bos),
            max_new_tokens,
            stop_ids,
            reset_cache=reset_cache,
            image=self.preprocess(images),
        )

    def stream(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        images=None,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[str]:
        """The completion as text deltas; `images` as in iter_tokens()."""
        tok = self._require_ready("stream")
        yield from _decode_stream(
            tok,
            self.iter_tokens(
                prompt,
                max_new_tokens,
                stop_ids,
                images=images,
                reset_cache=reset_cache,
                add_bos=add_bos,
            ),
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        images=None,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> str:
        """The completion of `prompt` (the new text only); `images` as in
        iter_tokens()."""
        return "".join(
            self.stream(
                prompt,
                max_new_tokens,
                stop_ids,
                images=images,
                reset_cache=reset_cache,
                add_bos=add_bos,
            )
        )

    def chat(
        self,
        *,
        template: ChatTemplate = GEMMA3_CHAT_TEMPLATE,
        max_new_tokens: int | None = None,
        image_dir: str | os.PathLike = ".",
        max_images_per_turn: int = 2,
    ) -> "Gemma3ChatSession":
        """Open a conversation on this model's KV cache (see ChatSession).
        Replies run until the model ends its turn unless `max_new_tokens` caps
        them; `template` is for a fine-tune trained with other pieces. On a
        VLM checkpoint a message may name images with <image:file> tags: each
        file is read from `image_dir` and stands where its tag stood, up to
        `max_images_per_turn` a turn."""
        return Gemma3ChatSession(
            self,
            template,
            max_new_tokens=max_new_tokens,
            image_dir=image_dir,
            max_images_per_turn=max_images_per_turn,
        )


class Gemma3ChatSession(ChatSession):
    """A ChatSession whose messages may carry <image:file> tags; see
    Gemma3.chat()."""

    def __init__(
        self,
        lm: Gemma3,
        template: ChatTemplate,
        max_new_tokens: int | None = None,
        image_dir: str | os.PathLike = ".",
        max_images_per_turn: int = 2,
    ):
        super().__init__(lm, template, max_new_tokens=max_new_tokens)
        self.image_dir = image_dir
        self.max_images_per_turn = max_images_per_turn

    def _take_images(self, user_message: str):
        """Each <image:file> becomes <start_of_image>, and its file one image
        of the turn's batch."""
        if not self._lm.has_vision:
            return user_message, None
        text, files = gemma3_take_image_tags(user_message)
        if not files:
            return text, None
        if len(files) > self.max_images_per_turn:
            raise ValueError(
                f"Gemma3ChatSession: at most {self.max_images_per_turn} images per turn, "
                f"got {len(files)}"
            )
        return text, self._lm.preprocess([os.path.join(self.image_dir, f) for f in files])
