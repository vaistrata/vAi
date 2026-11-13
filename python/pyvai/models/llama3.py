"""Llama 3 — high-level wrapper over the C++-bound model + tokenizer.

The model is `models/llama3/model.h` (C++); generate() / stream() /
iter_tokens() come from `_CausalLM`, and chat() opens a `ChatSession` on the
Llama 3 instruct template.
"""

from __future__ import annotations

from .._pyvai import Llama3Model, Llama3Config, Device, Dtype
from ._causal_lm import _CausalLM
from .chat import ChatSession, ChatTemplate


__all__ = ["Llama3", "Llama3Model", "Llama3Config", "LLAMA3_CHAT_TEMPLATE"]


# The Llama 3 instruct format, as tokenizer_config.json's chat_template
# renders it. generation_config.json names the three end-of-generation ids.
LLAMA3_CHAT_TEMPLATE = ChatTemplate(
    bos="<|begin_of_text|>",
    user_header="<|start_header_id|>user<|end_header_id|>\n\n",
    assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    end_of_turn="<|eot_id|>",
    eog_pieces=("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"),
)


class Llama3(_CausalLM):
    """Llama 3: bound C++ `Llama3Model` + `Tokenizer` with the completion API
    and chat(). `config` picks the checkpoint (default:
    `Llama3Config.preset("3.2-1B")`; see `Llama3Config.presets`)."""

    _bos_piece = "<|begin_of_text|>"
    _eos_pieces = ("<|end_of_text|>", "<|eot_id|>", "<|eom_id|>")

    def __init__(
        self,
        config: Llama3Config | None = None,
        w_default_dtype: Dtype = Dtype.F16,
        o_default_dtype: Dtype = Dtype.F32,
        device: Device | None = None,
        max_runtime_context: int | None = 4096,
        prefill_chunk: int = 0,
        kv_cache_dtype: Dtype = Dtype.F16,
    ):
        self.config = config if config is not None else Llama3Config.preset("3.2-1B")
        super().__init__(
            Llama3Model(
                self.config,
                w_default_dtype=w_default_dtype,
                o_default_dtype=o_default_dtype,
                device=device,
                max_runtime_context=max_runtime_context,
                prefill_chunk=prefill_chunk,
                kv_cache_dtype=kv_cache_dtype,
            )
        )

    def chat(
        self, *, template: ChatTemplate = LLAMA3_CHAT_TEMPLATE, max_new_tokens: int | None = None
    ) -> ChatSession:
        """Open a conversation on this model's KV cache (see ChatSession).
        Replies run until the model ends its turn unless `max_new_tokens` caps
        them; `template` is for a fine-tune trained with other pieces."""
        return ChatSession(self, template, max_new_tokens=max_new_tokens)
