# GPT-2 Benchmarks

## Summary

Runs GPT-2 under vAi and llama.cpp on the same GPU and compares prefill / decode throughput.

- Checkpoint: the same for both; llama.cpp reads the [f16 GGUF](#model-preparation) converted from it.
- Conditions, and where the frameworks differ: [benchmarks/README.md](../README.md#llm-benchmarks). Their command submission differs, which may affect decode.

> [!IMPORTANT]
> vAi holds GPT-2's linear biases and position embedding in FP16, where llama.cpp's GGUF keeps them in F32. Tested to cost no precision: in a PyTorch FP32 forward this moves the output about a thousandth as much as the F16 weights both frameworks share, and both pick the same top-1 token at every position checked.

## Model preparation

Begin once the GPT-2 checkpoint (`data/tokenizers/gpt2/tokenizer.json` + `data/ckpts/gpt2/openai/<size>/model.safetensors`) is in place, taken from the size's repository listed in [openai-weights.h](../../models/production/gpt2/openai-weights.h). The vAi bench needs only these files.

The llama.cpp bench reads `data/benchmarks/gpt2/<size>/model.gguf`, converted from the checkpoint as in [SETUP.md](../SETUP.md#llamacpp). The converter also needs these files in the checkpoint folder: the tokenizer, copied from where vAi keeps it, and the rest from the same repository:

```bash
cp data/tokenizers/gpt2/tokenizer.json data/ckpts/gpt2/openai/<size>/
curl -fL -o data/ckpts/gpt2/openai/<size>/config.json https://huggingface.co/<repository>/resolve/main/config.json
```

> [!WARNING]
> llama.cpp's converter fails on this checkpoint with `Can not map tensor 'h.0.attn.bias'`. Remove the `.attn.bias` tensors from `model.safetensors` before converting; vAi does not need them.

## Building and running

Build as in [benchmarks/README.md](../README.md#building-and-running), then:

```bash
build/bin/bench-gpt2-vai
build/bin/bench-gpt2-llamacpp
```

*With no arguments a bench measures every prompt length in the tables below (128 256 512); pass lengths to measure only those (e.g. `bench-gpt2-vai 512`).*

Options:

- `--size <name>`: which checkpoint to measure. The name is the folder under `data/ckpts/gpt2/openai/` and `data/benchmarks/gpt2/`: `124M` (default) · `355M` · `774M` · `1.5B`. Pass the same name to both benches.

  ```bash
  build/bin/bench-gpt2-vai      --size 355M
  build/bin/bench-gpt2-llamacpp --size 355M
  ```

- `--showText`: print rep 0's generated text for the last length after the table (off by default, so the output stays just the table)
- `BENCH_VERBOSE=1` (environment, llama.cpp bench): show llama.cpp's own logs

## Results

Each cell is mean ± SD, as the bench prints it; the Vulkan driver is its name and version from `vulkaninfo --summary` (in `vulkan-tools` or the Vulkan SDK).

### GPT-2 124M

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
