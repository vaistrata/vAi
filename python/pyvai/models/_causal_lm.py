"""Completion over a KV-cached causal LM — the base of the GPT-2, Llama 3,
Gemma 3 and PaliGemma wrappers.

iter_tokens() yields token ids as they are sampled, stream() the same as text
deltas, and generate() the whole completion. One generation at a time per
model: two interleaved generators would share one KV cache.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Iterator
from pathlib import Path

from .._pyvai import Tensor, Tokenizer


class ContextFullError(ValueError):
    """The KV cache cannot take the prompt: what it holds plus the prompt
    exceeds the model's max_runtime_context. Nothing was prefilled."""


class _CausalLM:
    # Piece prepended to a prompt that opens a fresh context (None: the model
    # has no BOS), and the pieces that end a completion by default.
    _bos_piece: str | None = None
    _eos_pieces: tuple[str, ...] = ()

    def __init__(self, net):
        self.net = net
        self.tokenizer: Tokenizer | None = None
        self._weights_loaded = False

    # -- setup ---------------------------------------------------------------

    def load_weights(self, path: str | Path):
        """Load the checkpoint folder at `path` (every *.safetensors in it)."""
        self.net.load_weights(str(path))
        self._weights_loaded = True
        return self

    def load_tokenizer(self, tokenizer_json: str | Path):
        """Load the checkpoint's `tokenizer.json`."""
        self.tokenizer = Tokenizer(str(tokenizer_json))
        return self

    # -- context state -------------------------------------------------------

    @property
    def tokens_so_far(self) -> int:
        """Tokens the KV cache currently holds."""
        return self.net.tokens_so_far

    @property
    def context_budget(self) -> int:
        """The KV-cache depth (the model's max_runtime_context)."""
        return self.net.max_runtime_context

    @property
    def context_left(self) -> int:
        """Tokens that still fit in the KV cache."""
        return self.context_budget - self.tokens_so_far

    def reset_cache(self) -> None:
        """Rewind the KV cache to an empty context."""
        self.net.reset_cache()

    # -- internals -----------------------------------------------------------

    def _require_ready(self, who: str) -> Tokenizer:
        if self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__}.{who}: call load_tokenizer(...) first")
        if not self._weights_loaded:
            raise RuntimeError(f"{type(self).__name__}.{who}: call load_weights(...) first")
        return self.tokenizer

    def _piece_ids(self, pieces: Iterable[str]) -> set[int]:
        assert self.tokenizer is not None
        return {i for i in (self.tokenizer.piece_to_id(p) for p in pieces) if i != -1}

    def _encode(self, text: str, add_bos: bool) -> list[int]:
        assert self.tokenizer is not None
        ids = list(self.tokenizer.encode(text))
        if add_bos and self._bos_piece is not None:
            bos = self.tokenizer.piece_to_id(self._bos_piece)
            if bos != -1 and ids[:1] != [bos]:  # never double a BOS the text spelled out
                ids.insert(0, bos)
        return ids

    # A wrapper whose model takes images overrides both hooks: `image` is what
    # its prefill takes beside the ids, and _prefill_extra_tokens what that
    # costs beyond them.
    def _prefill(self, ids: list[int], image: Tensor | None) -> Tensor:
        if image is not None:
            raise TypeError(f"{type(self).__name__} takes no image")
        return self.net.prefill(Tensor.from_u32(ids))

    def _prefill_extra_tokens(self, image: Tensor | None) -> int:
        return 0

    def _iter_ids(
        self,
        ids: list[int],
        max_new_tokens: int | None,
        stop_ids: Iterable[int] | None,
        *,
        reset_cache: bool,
        image: Tensor | None = None,
    ) -> Iterator[int]:
        """The decode loop over an encoded prompt; see iter_tokens()."""
        if not ids:
            raise ValueError(f"{type(self).__name__}: the prompt encodes to no tokens")
        stop = set(stop_ids) if stop_ids is not None else self._piece_ids(self._eos_pieces)

        if reset_cache:
            self.net.reset_cache()
        # The core does not check the budget: refuse a prompt that overflows it,
        # and stop decoding when the cache is full.
        budget = self.context_budget
        prompt_len = len(ids) + self._prefill_extra_tokens(image)
        if self.tokens_so_far + prompt_len > budget:
            raise ContextFullError(
                f"{type(self).__name__}: the prompt is {prompt_len} tokens on top of "
                f"{self.tokens_so_far} cached, over the KV-cache budget of {budget} "
                "(max_runtime_context)"
            )

        token = self._prefill(ids, image)  # the model also samples the first token
        nxt = int(token.to_u32()[0])
        produced = 0
        while (max_new_tokens is None or produced < max_new_tokens) and nxt not in stop:
            yield nxt
            produced += 1
            if produced == max_new_tokens or self.tokens_so_far + 1 > budget:
                return
            token = self.net.decode(token)  # the same Tensor, advanced in place
            nxt = int(token.to_u32()[0])

    # -- completion ----------------------------------------------------------

    def iter_tokens(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[int]:
        """Prefill `prompt`, then yield sampled ids until `max_new_tokens`
        (None: no cap; 0 still prefills), a `stop_ids` token (default: the
        model's end-of-sequence pieces; neither yielded nor cached) or a full
        KV cache. `reset_cache=False` appends `prompt` to the current context.
        `add_bos` defaults to `reset_cache` and never doubles a BOS the prompt
        spells out. Raises ContextFullError, before touching the cache, when
        the prompt does not fit."""
        self._require_ready("iter_tokens")
        if add_bos is None:
            add_bos = reset_cache
        yield from self._iter_ids(
            self._encode(prompt, add_bos), max_new_tokens, stop_ids, reset_cache=reset_cache
        )

    def stream(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> Iterator[str]:
        """The completion as text deltas, never the prompt. Same arguments as
        iter_tokens()."""
        tok = self._require_ready("stream")
        yield from _decode_stream(
            tok,
            self.iter_tokens(
                prompt, max_new_tokens, stop_ids, reset_cache=reset_cache, add_bos=add_bos
            ),
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = 128,
        stop_ids: Iterable[int] | None = None,
        *,
        reset_cache: bool = True,
        add_bos: bool | None = None,
    ) -> str:
        """The completion of `prompt` (the new text only). Same arguments as
        iter_tokens()."""
        return "".join(
            self.stream(prompt, max_new_tokens, stop_ids, reset_cache=reset_cache, add_bos=add_bos)
        )


def _decode_stream(tokenizer: Tokenizer, ids: Iterable[int]) -> Iterator[str]:
    """Decode a token stream into text deltas without splitting a multi-byte
    character: one incremental UTF-8 decoder holds a partial character back
    until the token completing it arrives. Bytes that never complete become
    U+FFFD at the end."""
    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for i in ids:
        text = utf8.decode(tokenizer.decode_bytes([i]))
        if text:
            yield text
    tail = utf8.decode(b"", final=True)
    if tail:
        yield tail
