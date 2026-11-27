# Whisper

Whisper (Radford et al., 2022) is an encoder-decoder Transformer for speech recognition developed by OpenAI. A 30 s log-mel spectrogram passes through two convolutions and a bidirectional encoder. A causal decoder then predicts text tokens while cross-attending to the encoder states.

**References**: [Paper](https://arxiv.org/abs/2212.04356), [Checkpoint (HuggingFace)](https://huggingface.co/openai/whisper-tiny), [Repository](https://github.com/openai/whisper)

## Sizes

| Variant | Factory | `embedDim` | Encoder / decoder layers | `numMels` |
|---|---|---|---|---|
| `tiny` | `ModelConfig::tiny()` | 384 | 4 / 4 | 80 |
| `base` | `ModelConfig::base()` | 512 | 6 / 6 | 80 |
| `small` | `ModelConfig::small()` | 768 | 12 / 12 | 80 |
| `medium` | `ModelConfig::medium()` | 1024 | 24 / 24 | 80 |
| `large-v3` | `ModelConfig::largeV3()` | 1280 | 32 / 32 | 128 |
| `large-v3-turbo` | `ModelConfig::largeV3Turbo()` | 1280 | 32 / 4 | 128 |

Every size uses head dimension 64, a 1,500-position audio context and a 448-position text context. `ModelConfig::byName("base")` picks a size by its variant name.

## Build and run

Weights and tokenizer go under `data/`:

```bash
for v in tiny base; do
  mkdir -p data/ckpts/whisper/openai/$v data/tokenizers/whisper/$v
  curl -fL -o data/ckpts/whisper/openai/$v/model.safetensors https://huggingface.co/openai/whisper-$v/resolve/main/model.safetensors
  curl -fL -o data/tokenizers/whisper/$v/tokenizer.json https://huggingface.co/openai/whisper-$v/resolve/main/tokenizer.json
done
```

Then build and run the demo:

```bash
cmake -S . -B build
cmake --build build --target demo-whisper --config Release
./build/bin/demo-whisper                          # tiny on data/audio/jfk.wav
./build/bin/demo-whisper your.wav base           # any 16-bit PCM or 32-bit float WAV
```

The included `data/audio/jfk.wav` should transcribe to:

> And so my fellow Americans ask not what your country can do for you ask what you can do for your country.

## Graph

`encoder::Net` runs once per 30 s window and `decoder::Net` once per token. `whisper::Net` owns both nets and shares one pipeline cache between them.

- **Encoder.** `Conv1d(k=3, pad=1)` runs as a 2D `Convolution` over a `[numMels, 1, 3000]` image. The loader places each 1D kernel in the middle row of a 3x3 kernel whose other rows are zero. The blocks are `vai::group::Transformer<Classic, Separate>` with an all-zero causal mask, which makes them bidirectional. Each decoder layer's cross K / V projections run on the encoder output.
- **Decoder.** Each layer is causal self-attention with a KV cache, then cross-attention over the encoder's K / V, then an MLP. Cross-attention is `vai::node::Attention` with no KV cache and no causal mask: the K / V carry the encoder's length (1500) and every query attends all of them.
- **Accumulation.** `Linear` accumulates in fp32 by default. Whisper's activation outliers make fp16 sums lose accuracy, so no setting is needed.

## Limits

- Audio is cut into fixed 30 s windows, so a word on a boundary can be split.
- Decoding is greedy, with no timestamps, no token suppression and no temperature fallback. Repetitive or silent audio can produce repeated text up to the 224-token cap.
