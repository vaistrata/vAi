"""Chat over a KV-cached causal LM.

A `ChatSession` keeps the conversation in the model's KV cache: each turn
prefills only the text written since the last one, and the reply is sampled
on top, as in the C++ chat demos (examples/llama3/chat.cpp). The template is
plain data (`ChatTemplate`) spelled out per model, because vAi does not run
the Jinja `chat_template` a checkpoint declares.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._causal_lm import ContextFullError, _decode_stream

if TYPE_CHECKING:
    from ._causal_lm import _CausalLM


@dataclass(frozen=True)
class ChatTemplate:
    """The literal pieces a chat-tuned checkpoint was trained with.
    `eog_pieces` end an assistant turn: the session stops on any of them
    without feeding it back, and closes the turn with `end_of_turn` when the
    next user message arrives."""

    bos: str
    user_header: str
    assistant_header: str
    end_of_turn: str
    eog_pieces: tuple[str, ...]

    def user(self, text: str) -> str:
        return self.user_header + text + self.end_of_turn


class ChatSession:
    """A multi-turn conversation, opened by a wrapper's chat(); `messages`
    logs it as {"role", "content"} dicts. The session owns the model's KV
    cache while in use: run no completions on that model between turns, and
    keep one live session per model."""

    def __init__(self, lm: _CausalLM, template: ChatTemplate, max_new_tokens: int | None = None):
        tok = lm._require_ready("chat")
        self._lm = lm
        self._template = template
        self.max_new_tokens = max_new_tokens
        self._eog = lm._piece_ids(template.eog_pieces)
        missing = [p for p in template.eog_pieces if tok.piece_to_id(p) == -1]
        if missing:
            raise RuntimeError(
                f"ChatSession: the tokenizer has no piece {missing} — "
                "wrong tokenizer.json for this template?"
            )
        self.messages: list[dict[str, str]] = []
        self._started = False

    @property
    def template(self) -> ChatTemplate:
        return self._template

    @property
    def context_left(self) -> int:
        """Tokens that still fit in the KV cache."""
        return self._lm.context_left

    def reset(self) -> None:
        """Forget the conversation: empty the KV cache and the message log."""
        self._lm.reset_cache()
        self.messages = []
        self._started = False

    def _delta(self, user_message: str) -> str:
        """The text this turn adds to the cache: the BOS on the first turn,
        afterwards the close of the previous reply (its end-of-turn token was
        never fed back), then the user turn and the open assistant header."""
        t = self._template
        return (
            (t.end_of_turn if self._started else t.bos) + t.user(user_message) + t.assistant_header
        )

    def _take_images(self, user_message: str):
        """(text, images): the message as the model reads it and what its
        prefill takes beside the ids; overridden for a model that takes images."""
        return user_message, None

    def stream(self, user_message: str, max_new_tokens: int | None = None) -> Iterator[str]:
        """Send one user turn and yield the reply as text deltas, until an
        end-of-generation token, a full KV cache (see context_left) or
        `max_new_tokens` (default: the session's; None: no cap). Raises
        ContextFullError if the turn does not fit; a turn that fails before it
        reaches the cache leaves the session untouched."""
        tok = self._lm.tokenizer
        text, image = self._take_images(user_message)  # raises before the cache is touched
        if not self._started:
            self._lm.reset_cache()  # a fresh conversation starts from an empty cache
        ids = self._lm._encode(self._delta(text), add_bos=False)
        n = len(ids) + self._lm._prefill_extra_tokens(image)
        if n > self.context_left:
            raise ContextFullError(
                f"ChatSession: this turn is {n} tokens but only {self.context_left} "
                f"of {self._lm.context_budget} fit — the conversation is full"
            )
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        reply: list[str] = []
        cached = self._lm.tokens_so_far
        try:
            yield from _collect(
                reply,
                _decode_stream(
                    tok,
                    self._lm._iter_ids(
                        ids, max_new_tokens, self._eog, reset_cache=False, image=image
                    ),
                ),
            )
        finally:
            # Recorded once the turn reached the cache, even if the consumer stops
            # reading early: the next turn closes what was produced. A prefill
            # that raised left the cache, and so the session, as it was.
            if self._lm.tokens_so_far > cached:
                self._started = True
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": "".join(reply)})

    def send(self, user_message: str, max_new_tokens: int | None = None) -> str:
        """Send one user turn and return the whole reply."""
        return "".join(self.stream(user_message, max_new_tokens))


def _collect(into: list[str], pieces: Iterator[str]) -> Iterator[str]:
    for piece in pieces:
        into.append(piece)
        yield piece
