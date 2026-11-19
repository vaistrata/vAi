#!/usr/bin/env python3
"""Export the MIGraphX bench's pi05 LIBERO bundle: static prefill and one-step ONNX graphs.

Consumers run prefill once and the step once per denoise step. The graphs are
FP16: parameters, activations, attention and the K/V cache passed from prefill to
the step. The RoPE frequencies (buffers), the model code's own FP32 islands
(RMSNorm variance, softmax), the images, the timestep, x and the Euler update stay
FP32. The split is checked in FP32 before the conversion. A temporary CPU
wrapper calls the repository utilities to preserve the vAi benchmark's exact
JPEG processing and tokenization; no GPU runtime is needed here. The graphs are
checked against LeRobot's own sample_actions on those inputs; the bench's
accuracy reference is scripts/reference.py's, not written here.

Torch folds the graphs' constant subgraphs by default, transposing the weights
once at export: MIGraphX's MLIR GEMM path, the bench's default, runs that bundle
fastest. --no-constant-folding keeps the checkpoint's tensors as they are, which
its hipBLASLt path runs faster.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import onnx
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma

REPO_ROOT = Path(__file__).resolve().parents[3]

# The MIGraphX bench's configuration, vlaConfigs::lerobotLibero in common/vla-config.h: the LeRobot LIBERO
# checkpoint (lerobot/pi05_libero_base, in the 3B folder) with its own policy (the discretized state in the prompt), two cameras, a
# 50-action chunk and 10 denoise steps. The graphs keep these shapes.
SIZE = "3B"
CAMERAS = 2
CHUNK = 50
STEPS = 10

INPUT_NAMES = ["images", "image_masks", "tokens", "token_masks", "noise"]
ARTIFACTS = [
    "prefill.fp16.onnx",
    "prefill.fp16.weights.bin",
    "step.fp16.onnx",
    "step.fp16.weights.bin",
    "metadata.json",
    "config.txt",
]

# Temporary CPU glue only: all preprocessing algorithms stay in the existing
# repository utilities. This is not a second image/tokenizer implementation.
FIXTURE_CPP = r"""
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "models/production/pi05/processing.h"
#include "bench-inputs.h"
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
namespace pi05 = model::production::pi05;
template<class T> void save(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!file) throw std::runtime_error("cannot write " + path.string());
}
int main(int argc, char** argv) try
{
    static_assert(std::endian::native == std::endian::little);
    if (argc != 5) throw std::runtime_error("usage: prepare-inputs DATA_ROOT OUTPUT_DIR CAMERAS CHUNK");
    const std::filesystem::path data(argv[1]), output(argv[2]);
    const int cameras = std::stoi(argv[3]), chunk = std::stoi(argv[4]);
    if (cameras < 1 || cameras > int(std::size(vlaCameras)) || chunk < 1)
        throw std::runtime_error("invalid camera count or chunk");
    std::vector<float> pixels;
    for (int i = 0; i < cameras; ++i) {
        const std::vector<float> image = vai::utils::ImageProcessor((data / vlaCameras[i]).string())
            .letterbox(224, vai::utils::Align::Center, 0.0f).normalize(-1.0f, 1.0f).chw();
        pixels.insert(pixels.end(), image.begin(), image.end());
    }
    const auto prompt = pi05::makePrompt(vlaTask, makeVLAState(32));
    vai::utils::Tokenizer tokenizer((data / "tokenizers/paligemma/tokenizer.json").string());
    const auto ids = tokenizer.encode(prompt, true);
    if (ids.empty() || ids.size() > 200) throw std::runtime_error("expected 1..200 prompt tokens");
    save(output / "images.bin", pixels);
    save(output / "tokens.bin", std::vector<int64_t>(ids.begin(), ids.end()));
    save(output / "noise.bin", pi05::sampleNoise(size_t(chunk) * 32, vlaNoiseSeed));
    std::ofstream file(output / "prompt.txt");
    file << prompt;
    if (!file) throw std::runtime_error("cannot write prompt.txt");
    return 0;
}
catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
"""


_STANDARD_DOMAINS = {"", "ai.onnx"}
_NONDETERMINISTIC_OPS = {"Bernoulli", "Dropout", "Multinomial"}
_CONTROL_FLOW_OPS = {"If", "Loop", "Scan"}


def _subgraphs(node):
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            yield attribute.g
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            yield from attribute.graphs


def _captured_names(graph):
    # Conservatively protect every name mentioned inside a nested graph. This
    # keeps implicit outer-scope captures valid without changing nested scopes.
    names = {value.name for value in graph.output}
    for node in graph.node:
        names.update(node.input)
        for nested in _subgraphs(node):
            names.update(_captured_names(nested))
    return names


def canonicalize_graph(model):
    """Deduplicate deterministic common expressions in place; return pass stats.

    Node names and documentation do not affect expression identity. Attribute
    protobufs, input order and omitted optional outputs do. Random operations,
    custom domains and nodes containing subgraphs are never merged. Required
    graph-output names survive as Identity aliases.
    """
    graph = model.graph
    before = len(graph.node)
    output_names = {value.name for value in graph.output}
    protected_names = set()
    for node in graph.node:
        for nested in _subgraphs(node):
            protected_names.update(_captured_names(nested))

    aliases = {}
    expressions = {}
    retained = []
    removed = 0

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    for node in graph.node:
        rewritten = [resolve(name) for name in node.input]
        del node.input[:]
        node.input.extend(rewritten)
        eligible = (
            node.domain in _STANDARD_DOMAINS
            and not node.op_type.startswith("Random")
            and node.op_type not in _NONDETERMINISTIC_OPS | _CONTROL_FLOW_OPS
            and not any(_subgraphs(node))
            and not any(name in protected_names for name in node.output)
            # These aliases are part of the public boundary, including on a
            # second invocation of this pass.
            and not (
                node.op_type == "Identity" and any(name in output_names for name in node.output)
            )
        )
        if not eligible:
            retained.append(node)
            continue
        attributes = tuple(
            attribute.SerializeToString(deterministic=True)
            for attribute in sorted(node.attribute, key=lambda attribute: attribute.name)
        )
        key = (
            node.op_type,
            node.domain,
            attributes,
            tuple(node.input),
            len(node.output),
            tuple(bool(name) for name in node.output),
        )
        previous = expressions.get(key)
        if previous is None:
            expressions[key] = tuple(node.output)
            retained.append(node)
            continue
        for old_name, canonical_name in zip(node.output, previous, strict=True):
            if old_name and old_name != canonical_name:
                aliases[old_name] = canonical_name
        removed += 1

    output_aliases = 0
    for value in graph.output:
        canonical_name = resolve(value.name)
        if canonical_name != value.name:
            retained.append(onnx.helper.make_node("Identity", [canonical_name], [value.name]))
            output_aliases += 1
    del graph.node[:]
    graph.node.extend(retained)

    # Removed intermediate names no longer exist. Inference will reconstruct
    # their surviving producers' annotations during validation.
    surviving_info = [value for value in graph.value_info if value.name not in aliases]
    del graph.value_info[:]
    graph.value_info.extend(surviving_info)
    for annotation in graph.quantization_annotation:
        annotation.tensor_name = resolve(annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = resolve(parameter.value)
    return {
        "nodes_before": before,
        "nodes_after": len(graph.node),
        "removed_nodes": removed,
        "output_aliases": output_aliases,
    }


def _boundaries(values):
    result = {}
    for value in values:
        if not value.type.HasField("tensor_type"):
            raise ValueError(f"expected tensor boundary: {value.name}")
        tensor = value.type.tensor_type
        if tensor.elem_type == onnx.TensorProto.UNDEFINED or not tensor.HasField("shape"):
            raise ValueError(f"expected a typed, static boundary: {value.name}")
        shape = []
        for dimension in tensor.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                raise ValueError(f"expected positive static dimensions: {value.name}")
            shape.append(dimension.dim_value)
        result[value.name] = {
            "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": shape,
        }
    return result


def validate_model(path):
    """Strictly validate a saved static graph without loading external weights.

    Shape inference explicitly checks operator input types. The default ONNX
    checker omits that check and can accept invalid bool CumSum inputs.
    """
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    for node in model.graph.node:
        if (
            node.domain not in _STANDARD_DOMAINS
            or node.op_type in _CONTROL_FLOW_OPS
            or any(_subgraphs(node))
        ):
            raise ValueError(f"expected a standard static ONNX graph: {node.name or node.op_type}")
    inputs = _boundaries(model.graph.input)
    outputs = _boundaries(model.graph.output)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=True, data_prop=False
    )
    graph = inferred.graph
    tensor_types = {
        value.name: value.type.tensor_type
        for value in [*graph.input, *graph.output, *graph.value_info]
        if value.type.HasField("tensor_type")
    }
    types = {name: tensor.elem_type for name, tensor in tensor_types.items()}
    types.update({tensor.name: tensor.data_type for tensor in graph.initializer})
    types.update(
        {tensor.values.name: tensor.values.data_type for tensor in graph.sparse_initializer}
    )
    cumsum_types = Counter()
    for node in graph.node:
        if node.op_type == "CumSum":
            dtype = types.get(node.input[0], onnx.TensorProto.UNDEFINED)
            if dtype in (onnx.TensorProto.UNDEFINED, onnx.TensorProto.BOOL):
                raise ValueError(
                    f"expected a known numeric CumSum input: {node.name or node.output[0]}"
                )
            cumsum_types[onnx.TensorProto.DataType.Name(dtype)] += 1
    rotary = [node for node in graph.node if "/rotary_emb/" in node.name]
    if any(node.op_type == "MatMul" for node in rotary):
        raise ValueError(
            "RoPE frequency MatMul would be converted to FP16; re-export with FP32 Mul"
        )
    frequency_products = 0
    for node in rotary:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        operands = [tensor_types.get(name) for name in node.input]
        if any(
            tensor is None
            or tensor.elem_type != onnx.TensorProto.FLOAT
            or len(tensor.shape.dim) != 3
            for tensor in operands
        ):
            continue
        left, right = (
            [dimension.dim_value for dimension in tensor.shape.dim] for tensor in operands
        )
        if left[1] == 128 and right[:2] == [1, 1] and right[2] > 0:
            frequency_products += 1
    if frequency_products != 1:
        raise ValueError("expected one FP32 broadcast RoPE frequency product per graph")
    # Passing the path keeps external-data locations relative to the saved
    # graph. The checker and inference inspect tensor metadata, not the weights.
    onnx.checker.check_model(str(path), full_check=True)
    return {
        "node_count": len(graph.node),
        "cumsum_count": sum(cumsum_types.values()),
        "cumsum_input_types": dict(sorted(cumsum_types.items())),
        "rope_frequency_op": "Mul",
        "rope_frequency_dtype": "FLOAT",
        "inputs": inputs,
        "outputs": outputs,
    }


def attention_mask(padding, attention):
    counts = torch.cumsum(attention.to(torch.int64), dim=1)
    allowed = counts[:, None, :] <= counts[:, :, None]
    return allowed & padding[:, None, :] & padding[:, :, None]


def attention_bias(core, mask, dtype):
    """LeRobot's additive attention mask, in the model's type.

    Its FP32 -2.38e38 overflows FP16, so the FP16 model takes the type's lowest value.
    """
    if dtype == torch.float32:
        return core._prepare_attention_masks_4d(mask)
    zero = torch.zeros((), dtype=dtype)
    lowest = torch.full((), torch.finfo(dtype).min, dtype=dtype)
    return torch.where(mask[:, None, :, :], zero, lowest)


@contextmanager
def images_in_model_dtype(core):
    """Feed SigLIP the images in its own parameter type.

    LeRobot keeps its vision tower FP32 and casts every image to FP32. In the FP16
    model the tower is FP16 too, so the image follows the weights instead.
    """
    model = core.paligemma_with_expert

    def embed_image(image):
        dtype = next(model.paligemma.model.vision_tower.parameters()).dtype
        return model.paligemma.model.get_image_features(image.to(dtype)).pooler_output

    model.embed_image = embed_image
    try:
        yield
    finally:
        del model.embed_image


def cache_from_tensors(values):
    cache = DynamicCache()
    for layer in range(len(values) // 2):
        cache.update(values[2 * layer], values[2 * layer + 1], layer)
    return cache


@contextmanager
def materialized_kv_heads():
    """Keep repeated single-head K/V packed instead of using a stride-zero view."""
    original = modeling_gemma.repeat_kv

    def repeat(hidden, count):
        if count == 1:
            return hidden
        if hidden.shape[1] == 1:
            return hidden.repeat(1, count, 1, 1)
        return original(hidden, count)

    modeling_gemma.repeat_kv = repeat
    try:
        yield
    finally:
        modeling_gemma.repeat_kv = original


@contextmanager
def fp32_rope_frequencies():
    """Export RoPE's FP32 outer product as Mul, outside dot/conv FP16 conversion.

    Keep the original rotary forward, including its autocast exclusion and
    dynamic-frequency update. Its [B, D/2, 1] @ [B, 1, P] has no reduction;
    broadcast multiplication is the same calculation.
    """
    original = modeling_gemma.GemmaRotaryEmbedding.forward

    class OuterProduct(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.replacements = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if func in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__):
                left, right = args
                if (
                    kwargs
                    or left.ndim != 3
                    or right.ndim != 3
                    or left.shape[-1] != 1
                    or right.shape[-2] != 1
                    or left.dtype != torch.float32
                    or right.dtype != torch.float32
                ):
                    raise RuntimeError("expected the FP32 RoPE frequency outer product")
                self.replacements += 1
                return left * right
            return func(*args, **kwargs)

    def forward(self, *args, **kwargs):
        with OuterProduct() as mode:
            result = original(self, *args, **kwargs)
        if mode.replacements != 1:
            raise RuntimeError("expected exactly one RoPE frequency outer product")
        return result

    modeling_gemma.GemmaRotaryEmbedding.forward = forward
    try:
        yield
    finally:
        modeling_gemma.GemmaRotaryEmbedding.forward = original


class Prefill(nn.Module):
    def __init__(self, core, cameras):
        super().__init__()
        self.core = core
        self.cameras = cameras

    def forward(self, images, image_masks, tokens, token_masks):
        core = self.core
        embeddings, padding, attention = core.embed_prefix(
            [images[:, i] for i in range(self.cameras)],
            [image_masks[:, i] for i in range(self.cameras)],
            tokens,
            token_masks,
        )
        # Torch's bool cumsum returns int64, but the legacy ONNX exporter puts
        # its Cast after CumSum. Cast first so the ONNX operation is well typed.
        mask = attention_mask(padding, attention)
        positions = torch.cumsum(padding.to(torch.int64), dim=1) - 1
        _, cache = core.paligemma_with_expert.forward(
            attention_mask=attention_bias(core, mask, embeddings.dtype),
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        values = tuple(value for layer in cache.layers for value in (layer.keys, layer.values))
        return (padding, *values)


class Step(nn.Module):
    def __init__(self, core, steps):
        super().__init__()
        self.core = core
        self.dt = -1.0 / steps

    def forward(self, x_t, timestep, cache_mask, *values):
        core = self.core
        # x_t and the Euler update stay FP32; the model runs in its parameters' type.
        dtype = core.action_in_proj.weight.dtype
        embeddings, padding, attention, condition = core.embed_suffix(
            x_t.to(dtype), timestep.to(dtype)
        )
        suffix_length = padding.shape[1]
        prefix = cache_mask[:, None, :].expand(padding.shape[0], suffix_length, cache_mask.shape[1])
        suffix = attention_mask(padding, attention)
        mask = torch.cat((prefix, suffix), dim=2)
        offsets = torch.sum(cache_mask.to(torch.int64), dim=-1)[:, None]
        positions = offsets + torch.cumsum(padding.to(torch.int64), dim=1) - 1
        # Each invocation owns a fresh cache container. update() concatenates
        # into new tensors and leaves the shared prefill K/V inputs unchanged.
        outputs, _ = core.paligemma_with_expert.forward(
            attention_mask=attention_bias(core, mask, dtype),
            position_ids=positions,
            past_key_values=cache_from_tensors(values),
            inputs_embeds=[None, embeddings],
            use_cache=False,
            adarms_cond=[None, condition],
        )
        # LeRobot casts to FP32 before the projection; the FP16 model keeps its own type.
        hidden = outputs[1][:, -core.config.chunk_size :].to(dtype)
        velocity = core.action_out_proj(hidden).to(torch.float32)
        return x_t + self.dt * velocity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--checkpoint",
        help=f"LeRobot checkpoint directory or Hugging Face id (default: ckpts/pi05/lerobot/{SIZE})",
    )
    parser.add_argument(
        "--output", type=Path, help=f"bundle directory (default: benchmarks/pi05/{SIZE})"
    )
    parser.add_argument(
        "--revision", help="Hugging Face checkpoint revision; prefer an immutable commit"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--constant-folding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let torch fold constant subgraphs (transposed weights and the like) at export, "
        "the bundle the bench's default MLIR GEMM path runs fastest (default: on); "
        "--no-constant-folding keeps the checkpoint's tensors, for its hipBLASLt path",
    )
    parser.add_argument(
        "--cxx", default=os.environ.get("CXX", "c++"), help="C++23 input-helper compiler"
    )
    return parser.parse_args()


def check_cpp23_compiler(command):
    guidance = "Set CXX or --cxx to a compiler with C++23 explicit object parameter support (for example, --cxx clang++)."
    try:
        compiler = shlex.split(command)
    except ValueError as error:
        raise RuntimeError(
            f"invalid C++ compiler command {command!r}: {error}. {guidance}"
        ) from error
    if not compiler:
        raise RuntimeError(f"empty C++ compiler command. {guidance}")
    source = """struct Probe {
    constexpr int value(this const Probe&) { return 0; }
};
static_assert(Probe{}.value() == 0);
"""
    try:
        result = subprocess.run(
            [*compiler, "-std=c++23", "-fsyntax-only", "-x", "c++", "-"],
            input=source,
            text=True,
            capture_output=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot run C++ compiler {command!r}: {error}. {guidance}") from error
    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"C++ compiler {command!r} failed the C++23 explicit object parameter probe. "
            f"{guidance}\n{details}"
        )
    return compiler


def export(args):
    compiler = check_cpp23_compiler(args.cxx)
    data = args.data_root.resolve()
    checkpoint = args.checkpoint or str(data / "ckpts/pi05/lerobot" / SIZE)
    output = (args.output or data / "benchmarks/pi05" / SIZE).resolve()
    if any(
        (output / name).exists()
        for name in [
            *ARTIFACTS,
            "model.fp32.onnx",
            "model.fp32.weights.bin",
            "prefill.fp32.onnx",
            "step.fp32.onnx",
        ]
    ):
        raise FileExistsError(f"an ONNX bundle already exists under {output}; choose --output")
    if not (data / "tokenizers/paligemma/tokenizer.json").is_file():
        raise FileNotFoundError(
            f"missing benchmark input: {data / 'tokenizers/paligemma/tokenizer.json'}"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("lerobot", "transformers", "torch", "onnx")
    }
    for name, expected in (
        ("lerobot", "0.6.0"),
        ("transformers", "5.5.4"),
        ("torch", "2.8.0"),
        ("onnx", "1.19.0"),
    ):
        if versions[name].split("+", 1)[0] != expected:
            raise RuntimeError(f"{name} {expected} required; found {versions[name]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".onnx-", dir=output.parent))
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="pi05-inputs-") as temporary:
            source, helper = Path(temporary) / "inputs.cpp", Path(temporary) / "inputs"
            source.write_text(FIXTURE_CPP)
            subprocess.run(
                [
                    *compiler,
                    "-std=c++23",
                    "-O2",
                    "-pthread",
                    "-I",
                    str(REPO_ROOT),
                    "-I",
                    str(REPO_ROOT / "vai/utils"),
                    "-I",
                    str(REPO_ROOT / "benchmarks/common"),
                    str(source),
                    str(REPO_ROOT / "vai/utils/image-processor.cpp"),
                    str(REPO_ROOT / "vai/utils/tokenizer.cpp"),
                    "-o",
                    str(helper),
                ],
                check=True,
            )
            subprocess.run(
                [str(helper), str(data), str(staging), str(CAMERAS), str(CHUNK)], check=True
            )

        import numpy as np
        from huggingface_hub import hf_hub_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from safetensors.torch import load_file

        options = {"revision": args.revision, "local_files_only": args.local_files_only}
        config = PreTrainedConfig.from_pretrained(checkpoint, **options)
        if not isinstance(config, PI05Config):
            raise RuntimeError(
                f"expected a pi05 checkpoint configuration; found {type(config).__name__}"
            )
        if (
            config.chunk_size != CHUNK
            or config.max_action_dim != 32
            or tuple(config.image_resolution) != (224, 224)
        ):
            raise RuntimeError(f"expected a pi05 [{CHUNK},32], 224x224 configuration")
        if config.paligemma_variant != "gemma_2b" or config.action_expert_variant != "gemma_300m":
            raise RuntimeError("expected the pi05 gemma_2b/gemma_300m architecture")
        if getattr(config, "rtc_config", None) is not None or getattr(
            config, "use_proprioceptive_memory", False
        ):
            raise RuntimeError("RTC and memory extensions are outside this benchmark")
        config.device, config.dtype, config.num_inference_steps = "cpu", "float32", STEPS
        config.compile_model, config.gradient_checkpointing = False, False
        policy = PI05Policy(config)
        weights = (
            Path(checkpoint) / "model.safetensors"
            if Path(checkpoint).is_dir()
            else Path(hf_hub_download(checkpoint, "model.safetensors", **options))
        )
        # Upstream from_pretrained catches loading failures; load strictly here.
        state = policy._fix_pytorch_state_dict_keys(load_file(str(weights)), config)
        state = {
            (key if key.startswith("model.") else "model." + key): value
            for key, value in state.items()
        }
        policy.load_state_dict(state, strict=True)
        del state
        core = policy.float().eval().model
        for module in core.modules():
            settings = getattr(module, "config", None)
            if settings is not None and hasattr(settings, "_attn_implementation"):
                settings._attn_implementation = "eager"
        if any(parameter.dtype != torch.float32 for parameter in core.parameters()):
            raise RuntimeError("expected FP32 model parameters")
        arrays = {
            "images": np.fromfile(staging / "images.bin", dtype="<f4").reshape(
                1, CAMERAS, 3, 224, 224
            ),
            "image_masks": np.ones((1, CAMERAS), dtype=np.bool_),
            "tokens": np.fromfile(staging / "tokens.bin", dtype="<i8").reshape(1, -1),
            "noise": np.fromfile(staging / "noise.bin", dtype="<f4").reshape(1, CHUNK, 32),
        }
        arrays["token_masks"] = np.ones(arrays["tokens"].shape, dtype=np.bool_)
        inputs = tuple(torch.from_numpy(arrays[name]) for name in INPUT_NAMES)
        step_options = (
            {"num_steps": STEPS}
            if "num_steps" in inspect.signature(core.sample_actions).parameters
            else {}
        )
        prefill, step_model = Prefill(core, CAMERAS).eval(), Step(core, STEPS).eval()
        timesteps = np.asarray([1.0 + i * (-1.0 / STEPS) for i in range(STEPS)], dtype="<f4")
        with torch.no_grad():
            print("Computing original FP32 PyTorch reference...", flush=True)
            reference = core.sample_actions(
                [inputs[0][:, i] for i in range(CAMERAS)],
                [inputs[1][:, i] for i in range(CAMERAS)],
                inputs[2],
                inputs[3],
                noise=inputs[4].clone(),
                **step_options,
            )
            if (
                reference.shape != (1, CHUNK, 32)
                or reference.dtype != torch.float32
                or not torch.isfinite(reference).all()
            ):
                raise RuntimeError("invalid reference action chunk")
        graph_metadata = {}

        def export_graph(model, example, names_in, names_out, label):
            path = staging / (label + ".fp16.onnx")
            print(f"Exporting {label} graph...", flush=True)
            # Without folding, the protobuf stays small and the checkpoint tensors are attached
            # by name. Folding computes new constants (transposed weights, constant
            # subgraphs), so torch writes them itself, per tensor once over 2 GB.
            fold = args.constant_folding
            torch.onnx.export(
                model,
                example,
                str(path),
                input_names=names_in,
                output_names=names_out,
                opset_version=18,
                dynamo=False,
                export_params=fold,
                do_constant_folding=fold,
                keep_initializers_as_inputs=not fold,
            )
            graph = onnx.load(str(path), load_external_data=False)
            parameters = dict(model.named_parameters(remove_duplicate=False))
            tensors = dict(parameters)
            tensors.update(dict(model.named_buffers(remove_duplicate=False)))
            external_path = staging / (label + ".fp16.weights.bin")
            tensor_counts = Counter()

            # All tensors go into one weights file, in the graph's order.
            def append(external, name, array):
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    raise RuntimeError(f"non-finite values in {name}")
                if array.dtype.byteorder == ">":
                    array = array.byteswap().view(array.dtype.newbyteorder("<"))
                array = np.ascontiguousarray(array)
                offset = external.tell()
                external.write(memoryview(array).cast("B"))
                tensor = onnx.TensorProto(
                    name=name,
                    dims=array.shape,
                    data_type=onnx.helper.np_dtype_to_tensor_dtype(array.dtype),
                )
                tensor.data_location = onnx.TensorProto.EXTERNAL
                for key, text in (
                    ("location", external_path.name),
                    ("offset", str(offset)),
                    ("length", str(array.nbytes)),
                ):
                    item = tensor.external_data.add()
                    item.key, item.value = key, text
                return tensor

            with external_path.open("wb") as external:
                if fold:
                    # Torch's initializers, inline or one file each: repacked, files removed as read.
                    packed = []
                    for initializer in graph.graph.initializer:
                        array = onnx.numpy_helper.to_array(initializer, base_dir=str(staging))
                        sources = {
                            item.value
                            for item in initializer.external_data
                            if item.key == "location"
                        }
                        tensor_counts[f"initializers_{array.dtype.name}"] += 1
                        packed.append(append(external, initializer.name, array))
                        del array
                        for source in sources:
                            (staging / source).unlink(missing_ok=True)
                    del graph.graph.initializer[:]
                    graph.graph.initializer.extend(packed)
                else:
                    # Parameters are FP16 like the whole model; buffers (RoPE frequencies)
                    # keep their type.
                    for value in graph.graph.input:
                        if value.name in names_in:
                            continue
                        if value.name not in tensors:
                            raise RuntimeError(
                                f"unknown ONNX parameter {value.name}; cannot attach checkpoint"
                            )
                        array = tensors[value.name].detach().cpu().contiguous().numpy()
                        if value.name in parameters and array.dtype != np.float16:
                            raise RuntimeError(f"expected an FP16 parameter: {value.name}")
                        kind = "parameters" if value.name in parameters else "buffers"
                        tensor_counts[f"{kind}_{array.dtype.name}"] += 1
                        graph.graph.initializer.append(append(external, value.name, array))
            graph_inputs = [value for value in graph.graph.input if value.name in names_in]
            del graph.graph.input[:]
            graph.graph.input.extend(graph_inputs)
            if [value.name for value in graph.graph.input] != names_in:
                raise RuntimeError(f"unexpected {label} inputs")
            if [value.name for value in graph.graph.output] != names_out:
                raise RuntimeError(f"unexpected {label} outputs")
            # FP32 images, timestep and x; the K/V cache between the graphs is FP16.
            for value in [*graph.graph.input, *graph.graph.output]:
                if value.name in ("image_masks", "token_masks", "cache_mask"):
                    dtype = onnx.TensorProto.BOOL
                elif value.name == "tokens":
                    dtype = onnx.TensorProto.INT64
                elif value.name[:1] in ("K", "V") and value.name[1:].isdigit():
                    dtype = onnx.TensorProto.FLOAT16
                else:
                    dtype = onnx.TensorProto.FLOAT
                if value.type.tensor_type.elem_type != dtype:
                    raise RuntimeError(f"incorrect ONNX boundary dtype: {value.name}")
            onnx.save_model(graph, str(path))
            validate_model(path)
            canonical = canonicalize_graph(graph)
            onnx.save_model(graph, str(path))
            checked = validate_model(path)
            graph_metadata[label] = {
                "canonicalization": canonical,
                "validation": checked,
                "tensors": dict(sorted(tensor_counts.items())),
            }
            print(f"{label}: {len(graph.graph.node)} nodes; strict ONNX checks passed", flush=True)

        with torch.no_grad(), materialized_kv_heads(), fp32_rope_frequencies():
            print("Checking split prefill/step against the original PyTorch model...", flush=True)
            cache = prefill(*inputs[:4])
            layers, prefix_length = (len(cache) - 1) // 2, cache[0].shape[1]
            if layers != 18 or any(
                tuple(value.shape) != (1, 1, prefix_length, 256) for value in cache[1:]
            ):
                raise RuntimeError("unexpected pi05 K/V cache shape")
            current, baseline = inputs[4].clone(), inputs[4].clone()
            step_errors = []
            for time_value in timesteps:
                timestep = torch.tensor([float(time_value)], dtype=torch.float32)
                velocity = core.denoise_step(
                    cache[0], cache_from_tensors(cache[1:]), baseline, timestep
                )
                baseline = baseline + (-1.0 / STEPS) * velocity
                current = step_model(current, timestep, *cache)
                torch.testing.assert_close(current, baseline, rtol=1e-5, atol=2e-5)
                step_errors.append(float((current - baseline).abs().max()))
            torch.testing.assert_close(current, reference, rtol=1e-5, atol=2e-5)
            split_error = float((current - reference).abs().max())
            print(
                f"Split CPU reference: max_abs={split_error:.6g}; all {STEPS} steps matched",
                flush=True,
            )

            # The checked split, now in FP16: parameters, activations and the K/V cache.
            # Buffers (the RoPE frequencies) stay FP32, and the model code's own FP32
            # islands (RMSNorm variance, softmax) stay as they are.
            print("Converting the parameters to FP16...", flush=True)
            for parameter in core.parameters():
                parameter.data = parameter.data.to(torch.float16)
            with images_in_model_dtype(core):
                print("Running the FP16 split on the CPU...", flush=True)
                cache = prefill(*inputs[:4])
                if any(value.dtype != torch.float16 for value in cache[1:]):
                    raise RuntimeError("expected an FP16 K/V cache")
                current = inputs[4].clone()
                for time_value in timesteps:
                    timestep = torch.tensor([float(time_value)], dtype=torch.float32)
                    current = step_model(current, timestep, *cache)
                if current.dtype != torch.float32 or not torch.isfinite(current).all():
                    raise RuntimeError("invalid FP16 action chunk")
                fp16_error = float((current - reference).abs().max())
                print(f"FP16 split vs FP32 reference: max_abs={fp16_error:.6g}", flush=True)
                kv_names = [name + str(layer) for layer in range(layers) for name in ("K", "V")]
                export_graph(
                    prefill, inputs[:4], INPUT_NAMES[:4], ["cache_mask", *kv_names], "prefill"
                )
                export_graph(
                    step_model,
                    (inputs[4], torch.ones(1, dtype=torch.float32), *cache),
                    ["x_t", "timestep", "cache_mask", *kv_names],
                    ["x_next"],
                    "step",
                )
        metadata = {
            "checkpoint": checkpoint,
            "revision": args.revision,
            "versions": versions,
            "cameras": CAMERAS,
            "chunk": CHUNK,
            "steps": STEPS,
            "seed": 0,
            "prompt": (staging / "prompt.txt").read_text(),
            "bundle_version": 5,
            "precision": "FP16 parameters, activations and K/V cache; FP32 images, timestep, x and Euler update",
            "cpu_fp16_max_abs": fp16_error,
            "constant_folding": args.constant_folding,
            "layers": layers,
            "prefix_length": prefix_length,
            "input_shapes": {name: list(arrays[name].shape) for name in INPUT_NAMES},
            "graphs": graph_metadata,
            "cpu_split_max_abs": split_error,
            "cpu_step_max_abs": step_errors,
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # bench::pi05::detail::readBundleConfig in migraphx/net.h reads it in this order.
        (staging / "config.txt").write_text(
            f"5 {CAMERAS} {STEPS} {CHUNK} 32 {layers} {prefix_length}\n"
        )
        output.mkdir(exist_ok=True)
        # Same-filesystem hard links publish without replacing existing files.
        # config.txt is last, so partial publication never looks ready.
        for name in ARTIFACTS:
            destination = output / name
            os.link(staging / name, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink()
        raise
    finally:
        shutil.rmtree(staging)
    print(f"Exported the MIGraphX bundle to {output}; ONNX checker passed.")


if __name__ == "__main__":
    export(parse_args())
