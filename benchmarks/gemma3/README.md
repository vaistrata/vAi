# Gemma 3 Benchmarks

## Summary

Runs Gemma 3 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: the same for both; llama.cpp reads the [f16 GGUF](#model-preparation) converted from it.
- Conditions, and where the frameworks differ: [benchmarks/README.md](../README.md#llm-benchmarks). Their command submission differs, which may affect decode.

## Model preparation

Begin once the Gemma 3 checkpoint (`data/tokenizers/gemma3/tokenizer.json` + `data/ckpts/gemma3/google/<size>/*.safetensors`, two shards for 4B) is in place, taken from the size's repository listed in [google-weights.h](../../models/production/gemma3/google-weights.h). The vAi bench needs only these files.

The llama.cpp bench reads `data/benchmarks/gemma3/<size>/model.gguf`, converted from the checkpoint as in [SETUP.md](../SETUP.md#llamacpp). The converter also needs these files in the checkpoint folder: the tokenizer, copied from where vAi keeps it, and the rest from the same repository:

```bash
cp data/tokenizers/gemma3/tokenizer.json data/ckpts/gemma3/google/<size>/
curl -fL -o data/ckpts/gemma3/google/<size>/config.json https://huggingface.co/<repository>/resolve/main/config.json
curl -fL -o data/ckpts/gemma3/google/<size>/tokenizer.model https://huggingface.co/<repository>/resolve/main/tokenizer.model
curl -fL -o data/ckpts/gemma3/google/<size>/tokenizer_config.json https://huggingface.co/<repository>/resolve/main/tokenizer_config.json
curl -fL -o data/ckpts/gemma3/google/<size>/special_tokens_map.json https://huggingface.co/<repository>/resolve/main/special_tokens_map.json
curl -fL -o data/ckpts/gemma3/google/<size>/added_tokens.json https://huggingface.co/<repository>/resolve/main/added_tokens.json
```

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then:

```bash
build/bin/bench-gemma3-vai
build/bin/bench-gemma3-llamacpp
```

*With no arguments a bench measures every prompt length in the tables below (128 256 512 1024 2048); pass lengths to measure only those (e.g. `bench-gemma3-vai 2048`).*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/ckpts/gemma3/google/` and `data/benchmarks/gemma3/`: `1B` (default) · `270M` · `4B`. Pass the same name to both benches.

  4B runs as text only: vAi loads its vision weights but never runs them, and llama.cpp's converter writes the language model alone, so the results do not measure image input.

  ```bash
  build/bin/bench-gemma3-vai      --size 270M
  build/bin/bench-gemma3-llamacpp --size 270M
  ```

- `--showText`: print rep 0's generated text for the last length after the table (off by default, so the output stays just the table)
- `BENCH_VERBOSE=1` (environment, llama.cpp bench): show llama.cpp's own logs

## Results

Each cell is mean ± SD, as the bench prints it; the Vulkan driver is its name and version from `vulkaninfo --summary` (in `vulkan-tools` or the Vulkan SDK).

### Gemma 3 270M

#### {GPU name}

| OS | Vulkan driver | GPU clock (core / memory) | vAi |
|---|---|---|---|
| {OS} | {driver} | {core} / {memory} | {commit} |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | - | - |
|  | llama.cpp | - | - |
| 256 | vAi | - | - |
|  | llama.cpp | - | - |
| 512 | vAi | - | - |
|  | llama.cpp | - | - |
| 1024 | vAi | - | - |
|  | llama.cpp | - | - |
| 2048 | vAi | - | - |
|  | llama.cpp | - | - |

### Gemma 3 1B

#### {GPU name}

| OS | Vulkan driver | GPU clock (core / memory) | vAi |
|---|---|---|---|
| {OS} | {driver} | {core} / {memory} | {commit} |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | - | - |
|  | llama.cpp | - | - |
| 256 | vAi | - | - |
|  | llama.cpp | - | - |
| 512 | vAi | - | - |
|  | llama.cpp | - | - |
| 1024 | vAi | - | - |
|  | llama.cpp | - | - |
| 2048 | vAi | - | - |
|  | llama.cpp | - | - |

### Gemma 3 4B (text)

#### {GPU name}

| OS | Vulkan driver | GPU clock (core / memory) | vAi |
|---|---|---|---|
| {OS} | {driver} | {core} / {memory} | {commit} |

| prompt | framework | prefill (t/s) | decode (t/s) |
|---:|---|---:|---:|
| 128 | vAi | - | - |
|  | llama.cpp | - | - |
| 256 | vAi | - | - |
|  | llama.cpp | - | - |
| 512 | vAi | - | - |
|  | llama.cpp | - | - |
| 1024 | vAi | - | - |
|  | llama.cpp | - | - |
| 2048 | vAi | - | - |
|  | llama.cpp | - | - |
