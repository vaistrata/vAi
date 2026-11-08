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
- `zimage-attention-i8` was then rewritten in the frame of `linear-q4-q8`: no shared memory, a 4-row x 4-key register tile per lane. Q·K uses `dotPacked4x8AccSatEXT` on int8 rows, which a small pass (`zimage-attention-i8-quant`) quantizes first, and P·V is summed in fp16. The old kernel made one shared read per int8 dot or per vec4 FMA, with no reuse across rows, and spent 15 ms on each half. A layer's attention takes 12.6 ms (0.8 ms quant + 11.8 ms) instead of 34 ms. rel L2 against fp32 is 3.99e-2 (old kernel 3.91e-2, cosine 0.99920 vs. 0.99924). Each 32-key tile is summed from zero and added to O once. Summing straight into an fp16 O gave 4.32e-2, and the fp16 registers are what let the 4-row tile fit: in fp32 it spills.

With the Adreno kernels that came after (`linear-q4-direct`, its int8 variant `linear-q4-q8`, and `zimage-attention-i8`), on the same phone and settings:

- `demo-zimage` samples 9 steps in 84 s. A step takes 5.6 s while the phone is cool and 9-13 s once it heats up. Before that, the weights load in 30 s and a warm-up forward takes 9 s.
- `bench-zimage-vai` gives rel L2 3.9e-2 against the fp32 forward (cosine 0.9992).
- The DiT loads from a prepared weights file, `data/zimage/<name>-q4-split.safetensors` for these settings (3.66 GB): each tensor already in the dtype the model runs, Q4 included. The first run converts the 23 GB fp32 checkpoint as before (36.5 s) and writes the file; later runs read it in 2.6 s. `bench-zimage-vai --prepared` loads the same way; the result is unchanged (rel L2 3.905e-2 either way).
- On an integrated GPU the prepared tensors are read straight into their device buffers, mapped (`NeuralNet::paramTensor`): the bytes are neither staged nor copied by the GPU. Reading and uploading the DiT took 2.6 s + 2.3 s and now takes 3.3 s. Steps run no slower from that memory (6230 vs 6233 ms). `demo-zimage` no longer runs a warm-up forward: step 0 uploads and records, and takes 6.0 s on a cool phone, the same as the other steps.

The VAE decoder runs on the GPU too (`models/zimage/vae.h`), so `demo-zimage` goes from caption features to `data/zimage/demo.png` on the phone. On the same phone, 512x512 from a 64x64 latent, against the fp32 diffusers decode of the same latent (`bench-zimage-vae-vai --nodes-per-submit 1`):

- rel L2 6.2e-5 (cosine 1.000000); the 8-bit images agree to 73.5 dB PSNR. Each stage (`conv_in`, mid block, up blocks) stays within 1.3e-4.
- A decode takes 2.25 s while the phone is cool (fp32 diffusers on an M-series CPU: 4.1 s). The 3x3 convs are 92% of it and run at about 1.1 TFLOPS; the mid-block attention over 4096 positions takes 36 ms.
- The decoder has its own conv node, `VaeConvNode`: on this GPU `vai::node::Convolution` returns wrong values for these shapes, through both its fallback and its Winograd plans.
- In one submission a decode outlasts the driver's submit timeout; with `--nodes-per-submit 1` its longest node takes 0.28 s.
- The conv's products and sums in fp16 (f16vec4 FMAs, flushed into fp32 every 1, 4 or 16 input channels) ran at 600 GFLOPS against fp32's 1,000 and dropped the decode to 61-66 dB PSNR, so the convs stay in fp32.

The caption encoder runs on the phone too (`models/zimage/text-encoder.h`): Qwen3-4B's first 35 decoder layers, whose output is the pipeline's `hidden_states[-2]`. The projections are Q4 and the norms fp32, 2.0 GB in all. The embedding table stays in its file as fp16 and only the prompt's rows are read. Against the fp32 forward of `torch/text_encoder.py` (`bench-zimage-text-encoder-vai --nodes-per-submit 1`, same phone):

- The prompt's token ids match transformers' chat template exactly. Seven prompts were checked: English, Korean, Japanese, emoji, combining accents, runs of spaces and tabs, and one cut off at 512 tokens.
- On the 20-token fox prompt, rel L2 is 4.98e-2 (cosine 0.99933). The error is the Q4 rounding's own: the fp32 forward with the same Q4 weights gives 4.84e-2, and after two layers vAi matches it exactly (8.201e-2). One token carries most of the norm. Over the others the error is 17%, and fp16 `down_proj` weights only bring that to 15% for 1.26 GB more.
- The DiT gives the same picture from these features as from the CUDA bf16 encoder's, with the same scene, layout and quality.
- Encoding takes 131 ms for 20 tokens and 2.2 s for 512. The weights load in 1.9 s.
- The prepared file (`data/zimage/Z-Image-Turbo-text-encoder-q4.safetensors`, 2.77 GB) has the projections in Q4 and the embedding table in fp16. The phone does not have room for the 8 GB checkpoint, so `tools.py text-encoder` writes the file on a desktop. Its bytes are identical to the file vAi writes (checked on a 2-layer cut).
- The sampler now defaults to 8 steps (`turboSigmas`, `demo-zimage --steps N`), as diffusers' `ZImagePipeline` example and ComfyUI use. The schedule is the same `linspace(1, 1/N, N)` under shift 3, only with N = 8. On the phone, 8 steps on the fox's noise give an image of the same quality as 9, with the same scene and detail, one step sooner.
- The Android app never holds the encoder and the DiT at the same time, so dropping a `NeuralNet` has to give its GPU memory back. It did not: `eva::Buffer` is a handle that frees nothing on its own, and `BufferPool` dropped its handles without destroying the buffers. The pool now destroys them when it goes. Each net owns its pool, and `~NeuralNet` waits for its GPU work first. With this change, the app's GPU memory goes from 6.2 GB to 4.3 GB.

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
- The VAE: `python3 benchmarks/zimage/torch/vae.py --stages` stores the fp32 decode of `data/zimage/demo-latent.bin` (or of a seeded latent), and `bench-zimage-vae-vai --stages --profile` compares against it and times the decode. Both take `--height` / `--width`.
- The caption encoder: `python3 benchmarks/zimage/torch/text_encoder.py --stages` stores the fp32 features of a prompt (`--check-hf` also compares transformers' bf16 run). `python3 examples/zimage/tools.py text-encoder` writes the prepared Q4 file, and `bench-zimage-text-encoder-vai` checks against the reference and times the encoder. `--layers N` runs and checks the first N layers. `--tokens N` times a prompt of N tokens instead.
- `--q4` stores the block projections as Q4; `--nodes-per-submit N` splits each forward into submissions of N nodes, for GPUs whose driver resets after a few seconds per submission (Adreno). `--split-ffn-up` splits the FFN up projection in two, for the same drivers. `demo-zimage` takes all three.
