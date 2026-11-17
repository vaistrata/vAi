# Llama 3 Benchmarks

## Summary

Runs Llama 3 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: the same for both; llama.cpp reads the [f16 GGUF](#model-preparation) converted from it.
- Conditions, and where the frameworks differ: [benchmarks/README.md](../README.md#llm-benchmarks). Their command submission differs, which may affect decode.

## Model preparation

Begin once the Llama 3 checkpoint (`data/tokenizers/llama3/tokenizer.json` + `data/ckpts/llama3/meta/<size>/*.safetensors`, two shards for 3.2-3B) is in place, taken from the size's repository listed in [meta-weights.h](../../models/production/llama3/meta-weights.h). The vAi bench needs only these files.

The llama.cpp bench reads `data/benchmarks/llama3/<size>/model.gguf`, converted from the checkpoint as in [SETUP.md](../SETUP.md#llamacpp). The converter also needs these files in the checkpoint folder: the tokenizer, copied from where vAi keeps it, and the rest from the same repository:

```bash
cp data/tokenizers/llama3/tokenizer.json data/ckpts/llama3/meta/<size>/
curl -fL -o data/ckpts/llama3/meta/<size>/config.json https://huggingface.co/<repository>/resolve/main/config.json
curl -fL -o data/ckpts/llama3/meta/<size>/tokenizer_config.json https://huggingface.co/<repository>/resolve/main/tokenizer_config.json
curl -fL -o data/ckpts/llama3/meta/<size>/special_tokens_map.json https://huggingface.co/<repository>/resolve/main/special_tokens_map.json
```

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then:

```bash
build/bin/bench-llama3-vai
build/bin/bench-llama3-llamacpp
```

*With no arguments a bench measures every prompt length in the tables below (128 256 512 1024 2048); pass lengths to measure only those (e.g. `bench-llama3-vai 2048`).*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/ckpts/llama3/meta/` and `data/benchmarks/llama3/`: `3.2-1B` (default) or `3.2-3B`. Pass the same name to both benches.

  ```bash
  build/bin/bench-llama3-vai      --size 3.2-3B
  build/bin/bench-llama3-llamacpp --size 3.2-3B
  ```

- `--showText`: print rep 0's generated text for the last length after the table (off by default, so the output stays just the table)
- `BENCH_VERBOSE=1` (environment, llama.cpp bench): show llama.cpp's own logs

## Results

Each cell is mean ± SD, as the bench prints it; the Vulkan driver is its name and version from `vulkaninfo --summary` (in `vulkan-tools` or the Vulkan SDK).

### Llama 3.2-1B

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

### Llama 3.2-3B

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
