"""GPT-2 — high-level wrapper over the C++-bound model + tokenizer.

The model is `models/production/gpt2/net.h` (C++); generate() / stream() /
iter_tokens() come from `_CausalLM`. GPT-2 has no chat template, so there is
no chat() here.
"""

from __future__ import annotations

from .._pyvai import GPT2Net, GPT2Config, Device, Dtype
from ._causal_lm import _CausalLM


__all__ = ["GPT2", "GPT2Net", "GPT2Config"]


class GPT2(_CausalLM):
    """GPT-2: bound C++ `GPT2Net` + `Tokenizer` with the completion API.
    `config` picks the checkpoint (default: `GPT2Config.preset("124M")`; see
    `GPT2Config.presets`)."""

    _bos_piece = None  # GPT-2 has no BOS
    _eos_pieces = ("<|endoftext|>",)

    def __init__(
        self,
        config: GPT2Config | None = None,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
        max_runtime_context: int | None = None,
        kv_cache_dtype: Dtype = Dtype.F16,
    ):
        self.config = config if config is not None else GPT2Config.preset("124M")
        super().__init__(
            GPT2Net(
                self.config,
                w_default_dtype=w_default_dtype,
                o_default_dtype=o_default_dtype,
                device=device,
                max_runtime_context=max_runtime_context,
                kv_cache_dtype=kv_cache_dtype,
            )
        )
