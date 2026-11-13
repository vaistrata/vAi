"""PaliGemma — high-level wrapper over the C++-bound vision-language model
+ tokenizer.

The model is `models/production/paligemma/net.h` (C++); generate() / stream() /
iter_tokens() come from `_CausalLM`, and here each takes the image its prompt
is about.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .._pyvai import PaliGemmaModel, PaliGemmaConfig, Device, Dtype, Resize, Tensor
from ._causal_lm import _CausalLM, _decode_stream
from ._image import image_processor


__all__ = ["PaliGemma", "PaliGemmaModel", "PaliGemmaConfig"]


class PaliGemma(_CausalLM):
    """PaliGemma: bound C++ `PaliGemmaModel` + `Tokenizer` with the completion
    API, where a prompt comes with an image. `config` picks the checkpoint
    (default: `PaliGemmaConfig.preset("3B-224")`; see `PaliGemmaConfig.presets`).
    The mix checkpoints answer task prefixes such as "caption en",
    "answer en <question>" or "detect <object>"."""

    _bos_piece = "<bos>"
    _eos_pieces = ("<eos>",)

    def __init__(
        self,
        config: PaliGemmaConfig | None = None,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
        max_runtime_context: int | None = 2048,
        kv_cache_dtype: Dtype = Dtype.F16,
    ):
        self.config = config if config is not None else PaliGemmaConfig.preset("3B-224")
        super().__init__(
            PaliGemmaModel(
                self.config,
                w_default_dtype=w_default_dtype,
                o_default_dtype=o_default_dtype,
                device=device,
                max_runtime_context=max_runtime_context,
                kv_cache_dtype=kv_cache_dtype,
            )
        )

    # -- image -----------------------------------------------------------------

    @property
    def image_size(self) -> int:
        """Side of the square the model takes; images are squashed to it."""
        return self.model.image_size

    def preprocess(self, image) -> Tensor:
        """An image (a file path, or an HWC uint8 RGB array) squashed to the
        model's square and scaled to [-1, 1]: the [3, S, S] float Tensor
        prefill takes. A Tensor of that shape passes through."""
        if isinstance(image, Tensor):
            return image
        return Tensor.from_numpy(
            image_processor(image)
            .resize(self.image_size, Resize.BILINEAR_ALIGN_CORNERS)
            .normalize(-1.0, 1.0)
            .chw()
            .to_numpy()
        )

    # -- prefill hooks -----------------------------------------------------------

    def _prefill(self, ids: list[int], image: Tensor | None) -> Tensor:
        return self.model.prefill(image, Tensor.from_u32(ids))

    def _prefill_extra_tokens(self, image: Tensor | None) -> int:
        return self.model.num_patches

    # -- completion about an image -----------------------------------------------

    def iter_tokens(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        image,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[int]:
        """As _CausalLM.iter_tokens; the required `image` (see preprocess())
        is prefilled ahead of `prompt` as num_patches soft tokens, which the
        budget check counts. The prompt gets the trailing newline the
        checkpoint was trained with when it lacks one."""
        self._require_ready("iter_tokens")
        if add_bos is None:
            add_bos = reset_cache
        if not prompt.endswith("\n"):
            prompt += "\n"
        yield from self._iter_ids(
            self._encode(prompt, add_bos),
            max_new_tokens,
            stop_ids,
            reset_cache=reset_cache,
            image=self.preprocess(image),
        )

    def stream(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        image,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[str]:
        """The completion as text deltas; `image` as in iter_tokens()."""
        tok = self._require_ready("stream")
        yield from _decode_stream(
            tok,
            self.iter_tokens(
                prompt,
                max_new_tokens,
                stop_ids,
                image=image,
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
        image,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> str:
        """The completion of `prompt` about `image` (the new text only)."""
        return "".join(
            self.stream(
                prompt,
                max_new_tokens,
                stop_ids,
                image=image,
                reset_cache=reset_cache,
                add_bos=add_bos,
            )
        )
