# Whisper

Whisper (Radford et al., 2022) is an encoder-decoder Transformer for speech recognition developed by OpenAI. A 30 s log-mel spectrogram passes through two convolutions and a bidirectional encoder; a causal decoder then predicts text tokens while cross-attending to the encoder states.

**References**: [Paper](https://arxiv.org/abs/2212.04356), [Checkpoint (HuggingFace)](https://huggingface.co/openai/whisper-tiny), [Repository](https://github.com/openai/whisper)

## Configuration

```cpp
auto modelCfg = whisper::ModelConfig::tiny();

whisper::Model model({
    .device        = device,
    .wDefaultDtype = Dtype::Fp16,
}, modelCfg);

loadWeights<"openai">(model);
```

### ModelConfig

| Factory | `variant` | `embedDim` | `numHeads` | encoder / decoder layers | `numMels` | `vocabSize` |
|---|---|---|---|---|---|---|
| `tiny()` | `"tiny"` | 384 | 6 | 4 / 4 | 80 | 51,865 |
| `base()` | `"base"` | 512 | 8 | 6 / 6 | 80 | 51,865 |
| `small()` | `"small"` | 768 | 12 | 12 / 12 | 80 | 51,865 |
| `medium()` | `"medium"` | 1024 | 16 | 24 / 24 | 80 | 51,865 |
| `largeV3()` | `"large-v3"` | 1280 | 20 | 32 / 32 | 128 | 51,866 |
| `largeV3Turbo()` | `"large-v3-turbo"` | 1280 | 20 | 32 / 4 | 128 | 51,866 |

`audioContext` 1,500 and `textContext` 448 are shared by all configurations. `ModelConfig::byName("base")` picks one by its variant name.

### RuntimeConfig

| Field | Default | |
|---|---|---|
| `kvCacheDtype` | `Fp16` | decoder self-attention cache |
| `linearAccDtype` | `Fp32` | accumulation of every `Linear` |

`Linear` otherwise accumulates in its weight dtype. With fp16 weights that loses Whisper's activation outliers: on tiny, the encoder's worst element error drops from 41% to 3% of the value range with fp32 accumulation.

## vAi graph

*`T`: number of input tokens, `D`: embedding dimension, `H`: number of heads, `Dh`: head dimension (64).*

Whisper runs as two `NeuralNet`s. The encoder runs once per 30 s window; the decoder once per token. The cross-attention K / V of every decoder layer are encoder outputs that the decoder takes as inputs. They never leave the GPU and keep their buffers, so the decoder does not re-record between windows. `whisper::Model` owns both nets and exposes `encode`, `prefill` and `decode`.

### Encoder

```cpp
input("inMel") - conv1 - conv2 - dropHeight - toTokens - "in0"/addPositions - transformer;
input("inPositions")                                   - "in1"/addPositions;

transformer - crossK[i] - splitK[i] - perK[i] - output("crossK" + i);   // per decoder layer
transformer - crossV[i] - splitV[i] - perV[i] - output("crossV" + i);
```

Whisper's `Conv1d(k=3, padding=1)` runs as the built-in 2D `Convolution` on a `[numMels, 1, 3000]` image: the loader places each 1D kernel in the middle row of a 3x3 kernel whose other rows are zero, and same-padding keeps the height at 1. The encoder layers are `vai::group::Transformer<Classic, Separate>` without a KV cache and with an all-zero causal mask, which makes them bidirectional.

### Decoder

```cpp
input("inTokenIds") - embedding - layers[0];
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
layers.back() - finalNorm - lastRow - lmHead - argmax - output("nextTokenId");

input("crossK" + i) - "crossK"/layers[i];
input("crossV" + i) - "crossV"/layers[i];
```

#### decoder::Layer

```cpp
input - selfNorm -            selfAttn - crossNorm - crossQ - splitQ - perQ - "Q"/crossAttn - mergeOut -            crossOut - ffnNorm -            ffn;
input            - "residual"/selfAttn                                                                 - "residual"/crossOut           - "residual"/ffn;
```

Each residual add is folded into the last projection of its branch through `acceptResidual`.

#### CrossAttention

`vai::node::Attention` requires K / V to carry the queries' own token count and appends them to its cache. Cross-attention reads a fixed set of `Tk` keys instead, so Whisper adds a node of its own in [`models/whisper/nodes/whisper-cross-attention.h`](../../../models/whisper/nodes/whisper-cross-attention.h) with the kernel [`whisper-cross-attention.comp`](../../../models/whisper/shaders/whisper-cross-attention.comp): one workgroup per (query, head), the scores of all `Tk` keys in shared memory, and a token-major `[T, H, Dh]` output that merges back with a reshape.

## Audio frontend

[`models/whisper/processing.h`](../../../models/whisper/processing.h) computes the log-mel spectrogram on the CPU, matching HuggingFace `WhisperFeatureExtractor`: a reflect-padded 400-point STFT (hop 160, periodic Hann window), Slaney mel filters over 0-8 kHz, `log10`, a clamp to 8 decades below the peak and `(x + 4) / 4`.

## Checkpoint

Each variant reads `data/whisper/ckpts/<variant>/model.safetensors` and `tokenizer.json` from `huggingface.co/openai/whisper-<variant>`. The demo looks special tokens up by name, as large-v3 shifts their ids.

## Demo

```bash
demo-whisper [audio.wav] [tiny|base|small|medium|large-v3|large-v3-turbo]
```

It reads 16-bit PCM or 32-bit float WAV at any sample rate (mono or mixed down), splits it into 30 s windows and transcribes each greedily with `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`. Windows are cut at fixed points, so a word on a boundary may split; timestamp-based seeking is not implemented. `data/whisper/jfk.wav` is included.

## Verification

Measured against HuggingFace `transformers` on `jfk.wav`:

| Check | Result |
|---|---|
| Log-mel (80 and 128 bins) | max abs. difference 3.7e-5 |
| Cross K / V, fp32 weights | max relative difference 5.8e-5 |
| Cross K / V, fp16 weights | max relative difference 3.2e-2, cosine 0.999998 |
| Greedy tokens, tiny and base | identical |
