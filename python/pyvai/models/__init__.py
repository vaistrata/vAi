"""pyvai built-in models, in two flavors.

A task wrapper (`GPT2`, `YOLOX`, ...) adds pre/post-processing to a bound
model and returns a task result: text, an action chunk, detections.
A bound model (`GPT2Model`, `YOLOXModel`, ...) is the C++ `Model` class
as-is: YOLOX is a `NeuralNet` called with a Tensor, the language
models, pi0 and pi0.5 are driven by `prefill()` and `decode()`. A checkpoint config
(`GPT2Config`, ...) is the model's C++ `ModelConfig`, built by `preset(variant)`.
"""

# Task wrappers
from .gpt2 import GPT2
from .llama3 import Llama3
from .gemma3 import Gemma3
from .paligemma import PaliGemma
from .pi0 import Pi0
from .pi05 import Pi05
from .yolox import YOLOX

# Bound models
from .gpt2 import GPT2Model
from .llama3 import Llama3Model
from .gemma3 import Gemma3Model
from .paligemma import PaliGemmaModel
from .pi0 import Pi0Model
from .pi05 import Pi05Model
from .yolox import YOLOXModel

# Supporting types: configs, enums, detection records. The tokenizer and a
# layer config's RopeConfig are engine-wide: pyvai.Tokenizer, pyvai.RopeConfig.
from .._pyvai import (
    TransformerLayerConfig,
    TransformerConfig,
    VisionModelConfig,
    LanguageModelConfig,
)
from .gpt2 import GPT2Config
from .llama3 import Llama3Config, LLAMA3_CHAT_TEMPLATE
from .gemma3 import Gemma3Config, Gemma3VisionEncoderConfig, GEMMA3_CHAT_TEMPLATE, Gemma3ChatSession
from .paligemma import PaliGemmaConfig
from .pi0 import Pi0Config, pi0_sample_noise
from .pi05 import Pi05Config, pi05_make_prompt
from .yolox import YOLOXTier, YOLOXDetection

# YOLO postprocessing, for driving YOLOXModel directly.
from .._pyvai import yolox_postprocess

# Gemma 3's prompt and image helpers (models/production/gemma3/processing.h), for Gemma3Model.
from .._pyvai import gemma3_expand_image_placeholders, gemma3_take_image_tags, gemma3_load_images

# Chat: the session and its template data.
from .chat import ChatSession, ChatTemplate
from ._causal_lm import ContextFullError


__all__ = [
    # Task wrappers
    "GPT2",
    "Llama3",
    "Gemma3",
    "PaliGemma",
    "Pi0",
    "Pi05",
    "YOLOX",
    # Bound models
    "GPT2Model",
    "Llama3Model",
    "Gemma3Model",
    "PaliGemmaModel",
    "Pi0Model",
    "Pi05Model",
    "YOLOXModel",
    # Supporting types
    "TransformerLayerConfig",
    "TransformerConfig",
    "VisionModelConfig",
    "LanguageModelConfig",
    "GPT2Config",
    "Llama3Config",
    "LLAMA3_CHAT_TEMPLATE",
    "Gemma3Config",
    "Gemma3VisionEncoderConfig",
    "GEMMA3_CHAT_TEMPLATE",
    "Gemma3ChatSession",
    "PaliGemmaConfig",
    "Pi0Config",
    "pi0_sample_noise",
    "Pi05Config",
    "pi05_make_prompt",
    "YOLOXTier",
    "YOLOXDetection",
    # Chat
    "ChatSession",
    "ChatTemplate",
    "ContextFullError",
    # YOLO postprocessing
    "yolox_postprocess",
    # Gemma 3 prompt and image helpers
    "gemma3_expand_image_placeholders",
    "gemma3_take_image_tags",
    "gemma3_load_images",
]
