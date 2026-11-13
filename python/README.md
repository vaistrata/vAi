# pyvai

[![status](https://img.shields.io/badge/status-experimental-orange)](#status--limitations)
[![python](https://img.shields.io/badge/python-3.9--3.14-blue)](#requirements)
[![vulkan](https://img.shields.io/badge/vulkan-1.3%2B-red)](#requirements)
[![platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%28experimental%29-lightgrey)](#status--limitations)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](../LICENSE)

<!-- Once published, swap the static badges above for dynamic ones:
![PyPI](https://img.shields.io/pypi/v/pyvai)
![Python versions](https://img.shields.io/pypi/pyversions/pyvai)
![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)
-->

## Overview

Python bindings for the **vAi** Vulkan inference engine, run the built-in models, or assemble your own graphs from the node DSL.

> [!WARNING]
> **`pyvai` is experimental.** The API can change between commits, there is no PyPI release, and the build-from-source install stays tied to the directory it was built in. See [Status / Limitations](#status--limitations).

## Installation

No PyPI release, build from source.

### Requirements

**System**:

- **Runtime**: a GPU driver that supports Vulkan 1.3 or newer (the API version the engine asks for)
- **Vulkan SDK**: 1.3.x or newer, for the headers, the loader and `glslc`, which compiles the shaders
- **Toolchain**: CMake ≥ 3.16, Ninja, a C++23 compiler (GCC 14 or newer: the core uses deducing `this`)

**Python**: 3.9 – 3.14. pip supplies the packages:

- **Build**: `scikit-build-core` + `nanobind`, fetched automatically. (Python 3.9 pins nanobind to 2.x; the marker in `pyproject.toml` does that for you.)
- **Runtime**: `numpy`, installed with the package. Images are decoded by the engine itself (`pyvai.ImageProcessor`).

### Install (editable)

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv --python 3.12
uv pip install -e .
```

Or with conda:

```bash
conda create -n pyvai python=3.12 && conda activate pyvai
pip install -e .
```

Where the default compiler is older than GCC 14, name the newer one for the build: `CC=gcc-14 CXX=g++-14 pip install -e .`.

> [!NOTE]
> If CMake stops at `Could NOT find Python (missing: Development.Module)`, the venv's base interpreter lacks the Python development headers. Install them (e.g. `sudo apt install python3.12-dev`) or pass `--managed-python` to `uv venv`.

### Docker

If you would rather work in a container, build the image from `docker/Dockerfile`. See [`docker/README.md`](../docker/README.md).

## Built-in Models

The C++ sample models come built in, each in two layers under `pyvai.models`: the **bound model** is the C++ `Model` class as-is (a `NeuralNet` built in C++, with the C++ weight loader and profile tags; YOLOX is called with a Tensor, the language models, pi0 and pi0.5 are driven by `prefill()` and `decode()`), and the **wrapper** adds what the C++ demo does around it (pre/post-processing, tokenizer, completion loop, chat).

| Model | Wrapper | Bound model | Config / helpers |
|---|---|---|---|
| GPT-2 | `GPT2` | `GPT2Model` | `GPT2Config` (presets `124M` … `1.5B`) |
| Llama 3 | `Llama3` | `Llama3Model` | `Llama3Config` (`3.2-1B`, `3.2-3B`), `LLAMA3_CHAT_TEMPLATE` |
| Gemma 3 | `Gemma3` | `Gemma3Model` | `Gemma3Config` (`270M`, `1B`, `4B` / `12B` VLMs), `GEMMA3_CHAT_TEMPLATE`, `gemma3_load_images` and the prompt helpers |
| PaliGemma | `PaliGemma` | `PaliGemmaModel` | `PaliGemmaConfig` (`3B-224`, `3B-448`) |
| pi0 | `Pi0` | `Pi0Model` | `Pi0Config` (`3B`), `pi0_sample_noise` |
| pi0.5 | `Pi05` | `Pi05Model` | `Pi05Config` (`3B`), `pi05_make_prompt` (noise: `pi0_sample_noise`) |
| YOLOX | `YOLOX` | `YOLOXModel` | `YOLOXTier`, `YOLOXDetection`, `yolox_postprocess` |

- **Configs**: each model's C++ `ModelConfig`, built by `preset(variant)` (see `presets`) from the shared `TransformerConfig` / `LanguageModelConfig` / `VisionModelConfig` types; nested edits land in place.
- **Runtime options** (LLMs): `max_runtime_context` (the KV-cache budget), `kv_cache_dtype` and, for Llama 3, `prefill_chunk`. The wrappers default to the C++ demos' values: a 4096-token budget (PaliGemma 2048, GPT-2 its trained 1024), the Gemma 3 4B VLM, and 128 new tokens. A bound model's budget defaults to the trained context, as the core's `RuntimeConfig`.
- **Weights**: `load_weights` takes the checkpoint folder, `data/<model>/ckpts/<variant>`.
- **Images**: a file path or an HWC uint8 array; `pyvai.ImageProcessor` decodes and resizes them as the C++ demos do, and prepares input for a bound model too.

## Run Examples

Each model below pairs its Python API with the matching runnable script in [`python/examples/`](examples/).

### GPT-2

```python
import pyvai
from pyvai.models import GPT2

gpt = GPT2()  # config=GPT2Config.preset("124M") by default
gpt.load_weights("data/gpt2/ckpts/124M")  # every *.safetensors in the folder
gpt.load_tokenizer("data/gpt2/tokenizer.json")  # HuggingFace tokenizer.json
text = gpt.generate("The future of artificial intelligence is", max_new_tokens=50)
print(text)  # the completion only, not the prompt

for piece in gpt.stream("Once upon a time", max_new_tokens=50):  # or token by token
    print(piece, end="", flush=True)
```

```bash
python python/examples/gpt2/demo.py --weights_dir data/gpt2/ckpts/124M --tokenizer_path data/gpt2/tokenizer.json
```

### Llama 3

```python
import pyvai
from pyvai.models import Llama3

llama = Llama3()  # 4096-token KV cache; None = trained context (128K)
llama.load_weights("data/llama3/ckpts/3.2-1B")
llama.load_tokenizer("data/llama3/tokenizer.json")
print(llama.generate("The capital of France is"))  # completion; stops at the model's EOS

chat = llama.chat()  # the conversation lives in the KV cache
while msg := input("> "):  # blank line ends the chat
    for piece in chat.stream(msg):  # each turn prefills only its delta
        print(piece, end="", flush=True)
    print()
```

```bash
# One-shot completion
python python/examples/llama3/demo.py --weights_dir data/llama3/ckpts/3.2-1B --tokenizer_path data/llama3/tokenizer.json

# Multi-turn chat
python python/examples/llama3/chat.py --weights_dir data/llama3/ckpts/3.2-1B --tokenizer_path data/llama3/tokenizer.json
```

`generate()` / `stream()` return only the new text and start a fresh context (`reset_cache=False` continues it).

### Gemma 3

```python
from pyvai.models import Gemma3, Gemma3Config

gemma = Gemma3(Gemma3Config.preset("1B"))  # text only; the default is the 4B VLM
gemma.load_weights("data/gemma3/ckpts/1B")  # google/gemma-3-1b-it (gated)
gemma.load_tokenizer("data/gemma3/tokenizer.json")
print(gemma.generate("The capital of France is"))

chat = gemma.chat()
while msg := input("> "):
    for piece in chat.stream(msg):
        print(piece, end="", flush=True)
    print()
```

```python
from pyvai.models import Gemma3

vlm = Gemma3()  # the 4B VLM, 4096-token KV cache
vlm.load_weights("data/gemma3/ckpts/4B")  # google/gemma-3-4b-it (gated)
vlm.load_tokenizer("data/gemma3/tokenizer.json")

print(vlm.generate("<start_of_image>Summarize the image(s).", images=["data/gemma3/captured0.png"]))

chat = vlm.chat(image_dir="data/gemma3")
print(chat.send("What is in <image:captured0.png>?"))
```

```bash
# The C++ demo's request: the 4B VLM summarizes data/gemma3/captured0.png
python python/examples/gemma3/demo.py --weights_dir data/gemma3/ckpts/4B --tokenizer_path data/gemma3/tokenizer.json

# Text only, on the 1B
python python/examples/gemma3/demo.py --preset 1B --weights_dir data/gemma3/ckpts/1B --tokenizer_path data/gemma3/tokenizer.json

# Multi-turn chat; a message carries <image:file> tags
python python/examples/gemma3/chat.py --weights_dir data/gemma3/ckpts/4B --tokenizer_path data/gemma3/tokenizer.json
```

On the 4B / 12B VLMs each `<start_of_image>` takes one image from `images=`, and a chat message names its images with `<image:file>` tags.

### PaliGemma

```python
from pyvai.models import PaliGemma

pg = PaliGemma()  # 2048-token KV cache; 8K trained
pg.load_weights("data/paligemma/ckpts/3B-224")  # google/paligemma-3b-mix-224 (gated)
pg.load_tokenizer("data/paligemma/tokenizer.json")

print(
    pg.generate("caption en", image="data/paligemma/dog.jpg")
)  # a task prefix: caption en, detect <object>, ...
```

```bash
python python/examples/paligemma/demo.py --weights_dir data/paligemma/ckpts/3B-224 --tokenizer_path data/paligemma/tokenizer.json --image_path data/paligemma/dog.jpg
```

### pi0

```python
from pyvai.models import Pi0

pi0 = Pi0()  # up to 3 cameras, 10 denoise steps
pi0.load_weights("data/pi0/ckpts/3B")  # lerobot/pi0_base
pi0.load_tokenizer("data/paligemma/tokenizer.json")  # PaliGemma's tokenizer

cams = [f"data/pi0/{v}.jpg" for v in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # normalized joint positions, zero-padded to 32
chunk = pi0.act(
    cams,
    "Take the shoe with patterned bottom from the table and place it on the mat",
    state,
    seed=0,
)  # [50, 32] float32
```

```bash
python python/examples/pi0/demo.py --weights_dir data/pi0/ckpts/3B --tokenizer_path data/paligemma/tokenizer.json --image_path data/pi0/cam_high.jpg data/pi0/cam_left_wrist.jpg data/pi0/cam_right_wrist.jpg
```

### pi0.5

```python
from pyvai.models import Pi05

pi05 = Pi05()  # up to 3 cameras, 10 denoise steps
pi05.load_weights("data/pi05/ckpts/3B")  # lerobot/pi05_base
pi05.load_tokenizer("data/paligemma/tokenizer.json")  # PaliGemma's tokenizer

cams = [f"data/pi0/{v}.jpg" for v in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # normalized to [-1, 1]; binned into the prompt
chunk = pi05.act(
    cams,
    "Take the shoe with patterned bottom from the table and place it on the mat",
    state,
    seed=0,
)  # [50, 32] float32
```

```bash
python python/examples/pi05/demo.py --weights_dir data/pi05/ckpts/3B --tokenizer_path data/paligemma/tokenizer.json --image_path data/pi0/cam_high.jpg data/pi0/cam_left_wrist.jpg data/pi0/cam_right_wrist.jpg
```

### YOLOX

```python
from pyvai.models import YOLOX, YOLOXTier

yolox = YOLOX(tier=YOLOXTier.LARGE)
yolox.load_weights(
    "data/yolox/ckpts/large"
)  # converted with models/experimental/yolox/megvii-weights-convertor.py
for d in yolox.detect("data/yolox/dog.jpg"):
    print(d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2)
```

```bash
python python/examples/yolox/demo.py --weights_dir data/yolox/ckpts/large --image_path data/yolox/dog.jpg --tier l
```

## Building Graphs

> [!WARNING]
> **This section is experimental.** The node DSL and the node signatures in `pyvai.nodes` follow the C++ core and can change between commits.

The graph builder is exposed as well: subclass `pyvai.NeuralNet`, instantiate nodes from `pyvai.nodes`, and wire them with the `-` operator (a mirror of vAi's C++ DSL) or with calls. You define the architecture and pick the weight key names yourself, where a built-in model fixes both in C++. Calling the net returns the output `Tensor` when the graph has one output (`net.output("out0")`), and a `{name: Tensor}` dict when it has several — the same rule Keras models follow.

Nodes attach to the net constructed last on the thread, so finish building one net before constructing the next; building inside the subclass's `__init__` does that by itself.

[`examples/mnist/main.py`](examples/mnist/main.py) is the worked example — a small MNIST convnet assembled from nodes, run on the checkpoint committed under `data/mnist/ckpts`. A hand-built graph loads its weights through a loader you write.

## Status / Limitations

`pyvai` is **experimental**; expect breaking changes between commits.

- **Build-from-source only**: install with `pip install -e .`; the result is not relocatable (the engine loads its compiled shaders from the build directory). Distributable wheels are not available in this experimental release.
- **Vulkan only**: no CPU fallback; the device is chosen at engine init.
- **Windows is experimental**: the bindings build and run on Windows, but Linux is where they are tested; expect rough edges there.
- **Weights are user-provided**: pyvai ships model code, not weights, and expects the C++ models' key naming (GPT-2/Llama 3/Gemma 3/PaliGemma/pi0/pi0.5 use the upstream HuggingFace checkpoints).

## License

[Apache 2.0](../LICENSE), same as the parent vAi repository. Model weights and external dependencies retain their respective licenses.
