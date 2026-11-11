"""Built-in models and their wrappers, without any checkpoint: boundaries,
presets, setup errors, YOLOX postprocessing, and the completion / chat logic
on a stub model and a byte-level stub tokenizer.
"""

import numpy as np
import pytest

import pyvai
from pyvai.models import (
    GPT2,
    GPT2Config,
    GPT2Net,
    Llama3Config,
    Llama3Net,
    Gemma3Config,
    Gemma3Net,
    PaliGemmaConfig,
    PaliGemmaNet,
    Pi0Config,
    Pi0Net,
    Pi05Config,
    Pi05Net,
    YOLOX,
    YOLOXNet,
    YOLOXTier,
    YOLOXDetection,
    yolox_postprocess,
    ChatSession,
    LLAMA3_CHAT_TEMPLATE,
)
from pyvai.models._causal_lm import _CausalLM


# ---------------------------------------------------------------------------
# Bound models — what the bindings declare about them
# ---------------------------------------------------------------------------


def test_all_models_declare_boundaries():
    """A bound NeuralNet subclass builds its graph in C++, out of sight of the
    Python input()/output() wrappers, so its binding must declare the names."""
    bound = [
        cls
        for name, cls in vars(pyvai._pyvai).items()
        if isinstance(cls, type) and issubclass(cls, pyvai.NeuralNet) and cls is not pyvai.NeuralNet
    ]
    assert {c.__name__ for c in bound} >= {
        "GPT2Net",
        "Llama3Net",
        "Gemma3Net",
        "PaliGemmaNet",
        "Pi0Net",
        "Pi05Net",
        "YOLOXNet",
    }
    for cls in bound:
        ins, outs = cls._pyvai_io
        assert ins and outs, f"{cls.__name__} declares no boundaries"
    for cls in (GPT2Net, Llama3Net):
        assert cls._pyvai_io == (("inTokenIds",), ("nextTokenId",))
    assert Gemma3Net._pyvai_io == (("inImages", "inTokenIds"), ("nextTokenId",))
    assert PaliGemmaNet._pyvai_io == (("inImage", "inTokenIds"), ("nextTokenId",))
    assert Pi0Net._pyvai_io == (
        ("inImages", "inPromptTokens", "inState", "inStateToken", "inActions"),
        ("outStateToken", "outActions"),
    )
    assert Pi05Net._pyvai_io == (("inImages", "inPromptTokens", "inActions"), ("outActions",))


@pytest.mark.parametrize(
    "cls, expected",
    [
        (GPT2Config, ["124M", "355M", "774M", "1.5B"]),
        (Llama3Config, ["3.2-1B", "3.2-3B"]),
        (Gemma3Config, ["270M", "1B", "4B", "12B"]),
        (PaliGemmaConfig, ["3B-224", "3B-448"]),
        (Pi0Config, ["3B"]),
        (Pi05Config, ["3B"]),
    ],
)
def test_config_presets(cls, expected):
    """preset(variant) covers exactly the variants `presets` lists and returns a
    fresh object each call, nested configs included; other names raise."""
    assert cls.presets == expected
    for variant in expected:
        assert cls.preset(variant).variant == variant
    a, b = cls.preset(expected[0]), cls.preset(expected[0])
    a.language_cfg.layer_cfg.embed_dim = 1
    assert a.language_cfg.layer_cfg.embed_dim == 1  # nested edits land in place
    assert b.language_cfg.layer_cfg.embed_dim != 1
    with pytest.raises(ValueError, match="unknown variant"):
        cls.preset("nope")


def test_model_setup_errors():
    """A missing checkpoint is a FileNotFoundError; the wrappers refuse a forward
    before load_weights(), which would die in the driver."""
    with pytest.raises(FileNotFoundError):
        YOLOXNet(YOLOXTier.SMALL).load_weights("does-not-exist")
    with pytest.raises(RuntimeError, match="load_weights"):
        YOLOX(tier=YOLOXTier.SMALL).detect(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="load_tokenizer"):
        GPT2(max_runtime_context=8).generate("x")


def test_yolox_postprocess_maps_boxes_back_and_checks_size():
    """One anchor: confidence = objectness * class score, the box mapped back to
    the source through the geometry and clipped; a bad row width is refused."""
    data = np.zeros(85, dtype=np.float32)
    data[:5] = [100.0, 60.0, 40.0, 20.0, 0.9]  # cx, cy, w, h, objectness
    data[5 + 16] = 0.8  # dog
    geometry = pyvai.ImageGeometry(source_width=100, source_height=50, scale_x=2.0, scale_y=2.0)
    dets = yolox_postprocess(data, geometry, 0.25, 0.45)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (40.0, 25.0, 60.0, 35.0)
    assert d.class_id == 16 and d.class_name == "dog" and abs(d.confidence - 0.72) < 1e-6
    assert isinstance(d, YOLOXDetection)
    assert yolox_postprocess(data, geometry, 0.75, 0.45) == []  # below threshold
    geometry.source_width = 45
    assert yolox_postprocess(data, geometry, 0.25, 0.45)[0].x2 == 45.0  # clipped to width
    geometry.pad_x = 20.0
    assert yolox_postprocess(data, geometry, 0.25, 0.45)[0].x1 == 30.0  # (80 - 20) / 2
    for bad in (np.zeros(84, dtype=np.float32), np.zeros(0, dtype=np.float32)):
        with pytest.raises(ValueError, match="multiple of 85"):
            yolox_postprocess(bad, geometry, 0.25, 0.45)
    geometry.scale_x = 0.0
    with pytest.raises(ValueError, match="scale"):
        yolox_postprocess(data, geometry, 0.25, 0.45)


# ---------------------------------------------------------------------------
# Completion and chat — the pure-Python layers, on stubs
#
# The stub tokenizer is byte-level: a text token is one UTF-8 byte (id < 256)
# and special pieces get ids from 256 up, concatenative like the real BPE ones.
# ---------------------------------------------------------------------------

_SPECIAL = {
    "<|begin_of_text|>": 256,
    "<|eot_id|>": 257,
    "<|end_of_text|>": 258,
    "<|eom_id|>": 259,
    "<|endoftext|>": 260,
}


class _StubTokenizer:
    def encode(self, text, add_special=False, parse_special=True):
        ids, i = [], 0
        while i < len(text):
            for piece, pid in _SPECIAL.items():
                if text.startswith(piece, i):
                    ids.append(pid)
                    i += len(piece)
                    break
            else:
                ids.extend(text[i].encode("utf-8"))
                i += 1
        return ids

    def decode_bytes(self, ids, skip_special=True):
        return bytes(i for i in ids if i < 256)

    def decode(self, ids, skip_special=True):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def piece_to_id(self, piece):
        return _SPECIAL.get(piece, -1)


class _StubModel:
    """Replays `script`, one id per prefill / decode, counting cached tokens as
    the C++ models do and logging every call's ids in `calls`."""

    def __init__(self, script, budget=64):
        self.script = list(script)
        self.max_runtime_context = budget
        self.tokens_so_far = 0
        self.calls = []

    def load_weights(self, path):
        pass

    def reset_cache(self):
        self.tokens_so_far = 0

    def _sample(self):
        return pyvai.Tensor.from_u32([self.script.pop(0) if self.script else ord("z")])

    def prefill(self, tokens):
        ids = tokens.to_u32().tolist()
        self.calls.append(ids)
        self.tokens_so_far += len(ids)
        return self._sample()

    def decode(self, token):
        self.calls.append(token.to_u32().tolist())
        self.tokens_so_far += 1
        return self._sample()


class _StubLM(_CausalLM):
    _bos_piece = "<|begin_of_text|>"
    _eos_pieces = ("<|eot_id|>", "<|end_of_text|>")


def _lm(script, budget=64):
    lm = _StubLM(_StubModel(script, budget))
    lm.load_weights("stub")
    lm.tokenizer = _StubTokenizer()
    return lm


def test_iter_tokens_semantics():
    """One prefill (with BOS on a fresh context) then one forward per token;
    max_new_tokens caps the count, 0 prefills only; reset_cache=False
    continues the context without a BOS; a stop id ends the run unyielded."""
    lm = _lm(list(b"abc") + [_SPECIAL["<|eot_id|>"], *b"xyz"])
    assert list(lm.iter_tokens("hi", max_new_tokens=2)) == list(b"ab")
    assert lm.net.calls == [[256, *b"hi"], [ord("a")]]  # BOS + prompt, then one token
    assert lm.tokens_so_far == 3 + 1

    lm.net.calls.clear()
    assert list(lm.iter_tokens("!", max_new_tokens=0, reset_cache=False)) == []
    assert lm.net.calls == [[ord("!")]] and lm.tokens_so_far == 5  # continued, no BOS

    assert list(lm.iter_tokens("?", max_new_tokens=None, reset_cache=False)) == []  # eot first
    assert lm.tokens_so_far == 6  # the stop id was not cached

    assert list(lm.iter_tokens("!", max_new_tokens=3)) == list(b"xyz")
    assert lm.net.calls[-1] == [ord("y")]  # no forward after the last yield


def test_chat_session_turns():
    """Each turn prefills only its delta on the continuing cache; a reply stops
    on an end-of-generation id, messages are logged, and reset() empties everything."""
    eot = _SPECIAL["<|eot_id|>"]
    lm = _lm([*b"Paris", eot, *b"Madrid", eot], budget=512)
    chat = ChatSession(lm, LLAMA3_CHAT_TEMPLATE)
    assert chat.max_new_tokens is None and chat.messages == []

    assert chat.send("capital of France?") == "Paris"
    first_prefill = lm.net.calls[0]
    assert first_prefill[0] == 256 and first_prefill.count(256) == 1  # BOS once, from the template
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    after_first = lm.tokens_so_far

    assert "".join(chat.stream("and Spain?", max_new_tokens=3)) == "Mad"
    assert lm.net.calls[-3][0] == eot  # the turn opened by closing the reply
    assert lm.tokens_so_far > after_first and chat.messages[-1]["content"] == "Mad"
    assert chat.context_left == 512 - lm.tokens_so_far

    chat.reset()
    assert chat.messages == [] and lm.tokens_so_far == 0
