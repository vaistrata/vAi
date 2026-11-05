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
- `bench-zimage-vai --scalar-attention` runs the scalar attention kernel even on a coopmat device.
