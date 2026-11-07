# Z-Image Benchmarks

## Summary

Runs one forward of the Z-Image-Turbo DiT (one denoising step) under vAi and PyTorch (diffusers) on the same inputs and compares latency and accuracy. The caption encoder and the VAE are outside the measurement.

- Checkpoint: `Tongyi-MAI/Z-Image-Turbo` for both — vAi reads the `transformer/` safetensors, diffusers loads the same folder.
- Conditions: as set out in the shared conditions of [benchmarks/README.md](../README.md#shared-conditions-vai-vs-diffusers).
- Precision: vAi runs fp16 weights and activations with fp32 accumulation in every Linear; PyTorch runs bf16. Both are scored against an fp32 diffusers forward.

## Results

NVIDIA A40 48GB (RunPod), driver 595.91.07, Ubuntu 24.04, Vulkan through the NVIDIA EGL ICD, GCC 14. PyTorch 2.9.1 + CUDA 12.8, diffusers 0.42.1. GPU clock not locked.

1024x1024 (latent 16x128x128, 4096 image tokens), 20 caption tokens from `tools.py prepare` (vAi pads them to 32), t = 0.5:

| framework | latency (ms) | rel L2 vs fp32 |
|---|---:|---:|
| vAi | 918 | 7.7e-4 |
| PyTorch bf16 eager | 747 | 1.3e-2 |
| PyTorch bf16 `torch.compile` | 562 | — |

GPU time per forward (`--profile` / `torch.profiler`):

| part | vAi (ms) | PyTorch eager (ms) |
|---|---:|---:|
| Linear (GEMM) | ~635 | ~390 |
| Attention | ~152 | ~80 |
| Norm and element-wise | ~92 | ~280 |

- vAi's fused norm / modulation / SwiGLU kernels make the element-wise part about 3x faster.
- The gap is in the GEMMs: `linear-cm` reaches 70–90 TFLOPS on the A40, against about 120 for cuBLAS, and no tile size closed it.
- The attention kernel is the model's own (`models/zimage/nodes/zimage-attention.h`). `vai::node::Attention` run cacheless under an all-bidirectional mask takes ~270 ms on the same forward.

### Accuracy across resolutions and caption lengths

Seeded Gaussian caption features (`--cap-tokens`), t = 0.5. Relative L2 error against the fp32 diffusers output.

| resolution | tokens (image + caption, padded) | vAi coopmat | vAi scalar | PyTorch bf16 | vAi latency (coopmat / scalar) |
|---|---|---:|---:|---:|---:|
| 768x1344 | 4032 + 96 (77 caption) | 1.23e-3 | 1.23e-3 | 2.13e-2 | 914 / 1913 ms |
| 1008x1008 | 4000 + 32 (1 caption, 3969 image) | 1.21e-3 | 1.19e-3 | 2.30e-2 | 892 / 1823 ms |
| 512x512 | 1024 + 32 (32 caption) | 1.30e-3 | 1.27e-3 | 2.17e-2 | 267 / 327 ms |
| 1280x720 | 3616 + 320 (300 caption) | 1.61e-3 | 1.60e-3 | 2.82e-2 | 878 / 1756 ms |
| 1024x1024 | 4096 + 32 (28 caption) | 1.26e-3 | 1.26e-3 | 2.24e-2 | 920 / 1917 ms |

The scalar column is the attention kernel for devices without cooperative matrices, forced on the A40 with `--scalar-attention`. Its 16-key tile (`scalarKeysPerTile`) comes from an Adreno 830 (Galaxy S25 Ultra): with 32 keys the kernel spills registers there and runs ~140x slower, past the driver's submit timeout at these lengths. Run alone on that GPU, the 16-key kernel matches a CPU reference to 2e-4.

### Q4 block weights

`RuntimeConfig::blockWeightDtype = Dtype::Q4` (`--q4` in the bench and the demo) stores every block's attention and FFN projections as Q4: 4-bit values in groups of 32 with an fp16 scale, quantized like ggml's Q4_0. Those projections hold 97% of the weights, so the DiT drops from ~12 GB to ~3.7 GB. The other weights stay fp16.

A40, same inputs and reference as above (28 random caption tokens), coopmat attention:

| resolution | fp16: rel L2 / latency | Q4: rel L2 / cosine / latency |
|---|---:|---:|
| 512x512 | 1.31e-3 / 267 ms | 2.25e-1 / 0.9747 / 266 ms |
| 1024x1024 | 1.26e-3 / 921 ms | 2.39e-1 / 0.9713 / 909 ms |

- The error is the quantization's own. The fp32 diffusers forward with its block weights put through the same Q4 rounding scores 2.253e-1 at 512x512, as vAi does.
- Other 4-bit roundings change it little: groups of 16 give 2.05e-1, scale and minimum per group (Q4_1-style) give 2.12e-1, and both together 1.64e-1.
- Despite the per-step error, the 9-step demo gives an image of the same quality, with the same scene and small changes in detail.
- Latency is unchanged on the A40: these GEMMs are not memory-bound at these lengths.

Galaxy S25 Ultra (Adreno 830), 512x512, `--q4 --nodes-per-submit 1 --split-ffn-up`, scalar attention and scalar GEMM tiles:

- The Q4 DiT fits in the phone's memory and runs end to end. A forward takes 120-135 s, depending on how hot the phone is. The first, which also uploads the weights and records the graph, takes 110-140 s.
- Against the fp32 diffusers forward with the same Q4 rounding, rel L2 is 3.1e-2 (cosine 0.9995). Stage by stage through `layers.6`, the error matches the A40's to three digits.
- `--split-ffn-up` runs the FFN's w1 and w3 as two Linears joined by a Concat. That halves the DiT's longest node, from ~1.1 s to ~0.55 s at this size. Without it, a hot phone runs that node past the driver's ~1.7 s submit timeout.
- Where packed int8 dot products are accelerated, as on this GPU, the attention runs `zimage-attention-i8`: Q·K on int8 (K quantized per key, the query per row) and V staged as fp16. On a cool phone it takes 34 ms per layer instead of the scalar kernel's 83 ms (N = 1056, 30 heads), and the first 4 layers plus one refiner pair take 1.65 s instead of 1.87 s. rel L2 against fp32 goes from 3.70e-2 to 3.91e-2. `--no-int8-attention` keeps the scalar kernel.

With the Adreno kernels that came after (`linear-q4-direct`, its int8 variant `linear-q4-q8`, and `zimage-attention-i8`), on the same phone and settings:

- `demo-zimage` samples 9 steps in 84 s. A step takes 5.6 s while the phone is cool and 9-13 s once it heats up. Before that, the weights load in 30 s and a warm-up forward takes 9 s.
- `bench-zimage-vai` gives rel L2 3.9e-2 against the fp32 forward (cosine 0.9992).
- The DiT loads from a prepared weights file, `data/zimage/<name>-q4-split.safetensors` for these settings (3.66 GB): each tensor already in the dtype the model runs, Q4 included. The first run converts the 23 GB fp32 checkpoint as before (36.5 s) and writes the file; later runs read it in 2.6 s. `bench-zimage-vai --prepared` loads the same way; the result is unchanged (rel L2 3.905e-2 either way).

## Building and running

Checkpoint: the full `Tongyi-MAI/Z-Image-Turbo` snapshot in `data/zimage/Z-Image-Turbo/`. The DiT loader reads only `transformer/`; `tools.py` reads the text encoder and the VAE from the same folder.

```bash
python3 examples/zimage/tools.py prepare "A red fox sitting in a snowy birch forest at dawn"
python3 benchmarks/zimage/torch/bench.py --compile      # stores the fp32 reference, times PyTorch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON
cmake --build build --config Release --target bench-zimage-vai
build/bin/bench-zimage-vai --profile                    # accuracy, vAi latency, GPU time per node kind
```

- `torch/bench.py --stages` with `bench-zimage-vai --stages` compares the activations stage by stage, to find where an error comes in.
- Another resolution: pass the same `--height` / `--width` to both benches. Another caption length: `torch/bench.py --cap-tokens N`.
- `bench-zimage-vai --scalar-attention` runs the scalar attention kernel even on a coopmat device, and `--no-int8-attention` keeps it in fp32 where the int8 kernel would run.
- `--q4` stores the block projections as Q4; `--nodes-per-submit N` splits each forward into submissions of N nodes, for GPUs whose driver resets after a few seconds per submission (Adreno). `--split-ffn-up` splits the FFN up projection in two, for the same drivers. `demo-zimage` takes all three.
