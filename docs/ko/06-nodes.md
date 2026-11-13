# 제공 노드들

빌트인 노드 29종. 개념은 [노드(Node)](03-usage/node.md)를, 새 노드를 만드는 법은 [커스텀 노드와 커널](contributing/nodes.md)을 참조하세요.

## 목록

| 노드 | 요약 | 커널 |
|---|---|---|
| [Activation](#activation) | Applies an element-wise activation. | `activation.comp` |
| [Add](#add) | element-wise addition for residual connections | `add.comp` |
| [Argmax](#argmax) | Row-wise argmax / greedy sampling | `argmax.comp` |
| [Attention](#attention) | fused KV cache append + causal attention (GQA), split between three kernels | `attention-fa-intel.comp`, `attention-fa-cm.comp`, `attention-fa.comp`, `attention-fa-reduce.comp`, `attention.comp` |
| [Concat](#concat) | join inputs along one axis | `concat.comp` |
| [Convolution](#convolution) | 2D convolution with device-selected direct, implicit, or Winograd plans | `conv-k1s1-gemm-cm.comp`, `conv-k1s1-gemm.comp`, `conv-k3s2-gemm-cm.comp`, `conv-k3s1-implicit-gemm-cm.comp`, `conv-k3s1-implicit-gemm.comp`, `conv-implicit-gemm.comp`, `conv-implicit-gemm-cm.comp`, `conv-k3s2-gemm.comp`, `cm-transform.comp`, `conv-k3s1-winograd-filter-transform.comp`, `conv-k3s1-winograd-input.comp`, `conv-k3s1-winograd-output.comp`, `bn-reduce.comp`, `conv-k3s1-winograd-gemm.comp`, `conv-k3s1-winograd-gemm-cm.comp` |
| [CyclicTableReader](#cyclictablereader) | read successive table rows, wrapping at the end | `cyclic-table-reader.comp` |
| [EleAffine](#eleaffine) | element-wise affine, O = scale * in0 + shift | `ele-affine.comp` |
| [Flatten](#flatten) | Flatten tensor to 1D | — |
| [Gate](#gate) | conditional branch (no computation) | — |
| [GatedMul](#gatedmul) | element-wise multiply with one operand activated (GLU gating) | `gated-mul.comp` |
| [GroupNorm](#groupnorm) | group normalization over a CHW feature map | `group-norm.comp` |
| [Identity](#identity) | Pass-through node (no computation) | — |
| [IndexCopy](#indexcopy) | write source slices at listed indices, as torch.index_copy | `index-copy.comp` |
| [LayerNorm](#layernorm) | layer normalization over the feature axis, single kernel | `layer-norm.comp` |
| [Linear](#linear) | O = I · Wᵀ, split between two output decompositions | `linear-pack-intel.comp`, `linear-slab.comp`, `linear-tile-intel.comp`, `linear-cm-nv.comp`, `linear-cm.comp`, `linear-tile.comp`, `linear-reduce.comp` |
| [Mul](#mul) | Element-wise multiplication | `mul.comp` |
| [NearestUpsample](#nearestupsample) | Replicates CHW pixels by an integer scale factor. | `upsample.comp` |
| [Permute](#permute) | Permutes the axes of a tensor of any rank. | `permute.comp` |
| [Pool2d](#pool2d) | 2D pooling with a square kernel (MaxPool2d, AvgPool2d) | `pool2d.comp` |
| [Reshape](#reshape) | Reshape tensor to specified dimensions | — |
| [RmsNorm](#rmsnorm) | RMS normalization over the feature axis, single kernel | `rms-norm.comp` |
| [Rope](#rope) | rotary position embedding (half-split rotation) | `rope.comp` |
| [RowSliceScatter](#rowslicescatter) | write slices into rows, as torch.slice_scatter on dim 0 | — |
| [Slice](#slice) | Take a run of consecutive indices along one axis | `slice.comp` |
| [TimeEmbedding](#timeembedding) | sinusoidal embedding of a timestep the node counts down itself | `time-embedding.comp` |
| [TokenEmbedding](#tokenembedding) | token embedding, optionally fused with positional embedding | `token-embedding.comp` |
| [TopKSampling](#topksampling) | row-wise top-k sampling, one kernel run as two passes | `topk-sampling.comp` |
| [TypeCast](#typecast) | Casts each tensor element to another dtype. | `cast.comp` |

---

## Activation

Applies an element-wise activation.

소스: [`vai/nodes/activation.h`](../../vai/nodes/activation.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: epilogue(in0); optionally in-place

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |
| `inPlace` | `bool` | `false` |  |

**동작 및 제약**

The inPlace option aliases the output to the input and requires matching input
and output dtypes. Otherwise, the output uses separate storage.

**커널**

- `vai/shaders/activation.comp`

---

## Add

element-wise addition for residual connections

소스: [`vai/nodes/add.h`](../../vai/nodes/add.h)

**시그니처**

- 입력: in0, in1 [same shape]
- 출력: in0 + in1

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**동작 및 제약**

Inputs must have identical shapes and dtypes.

**커널**

- `vai/shaders/add.comp`

---

## Argmax

Row-wise argmax / greedy sampling

소스: [`vai/nodes/argmax.h`](../../vai/nodes/argmax.h)

**시그니처**

- 입력: in0 [M, N] (fp32 logits)
- 출력: out0 [M] (one raw uint32 index per row, in 4-byte slots)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

Each row is reduced independently. out0[m] is the index of the largest value in
in0[m, :]. Output elements are raw uint32 values, not fp32 numeric values.

**커널**

- `vai/shaders/argmax.comp`

---

## Attention

fused KV cache append + causal attention (GQA), split between three kernels

소스: [`vai/nodes/attention.h`](../../vai/nodes/attention.h)

**시그니처**

- 입력: Q [..., Hq, Tq, headDim], K / V [..., Hkv, Tkv, headDim]
- 출력: out0 attention [..., Tq, Hq, headDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `K` | input |  |
| `Q` | input |  |
| `V` | input |  |
| `out0` | output |  |
| `K_cache` | persistent |  |
| `V_cache` | persistent |  |
| `partialML` | temporary |  |
| `partialO` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `headDim` | `uint32_t` | — |  |
| `numQHeads` | `uint32_t` | — |  |
| `numKVHeads` | `uint32_t` | `0` | for GQA (Hkv < numQHeads); 0 → Hkv = numQHeads |
| `kvCacheSize` | `uint32_t` | `0` | tokens the cache holds; 0 = no cache at all |
| `kvCacheDtype` | `vai::Dtype` | `vai::Dtype::None` | K / V cache; required when kvCacheSize > 0 |
| `kvCacheRing` | `bool` | `false` | the cache keeps the last kvCacheSize positions, rounded up to kvRingAlign |

**동작 및 제약**

Leading axes are more sequences swept in one dispatch; they ride the grid,
so every count below stays per-sequence.

Q, K, V arrive as three separate buffers (so callers can transform Q/K but
not V in between — e.g. RoPE). Supports GQA (numKVHeads < numQHeads).

- coopmat FA (attention-fa-cm): subgroup-MMA flash attention,
fixed 16x64 tile; prefill and GQA-packed decode on cm-capable devices.
- scalar FA (attention-fa): flash attention with per-(shape,
vendor) tuned tiles; everything the cm gates reject.
- Intel FA (attention-fa-intel): Intel parts without a coopmat path;
a 16-row tile packs one KV group's heads, subgroup-broadcast products.
- naive (attention): the unconditional fallback — unfamiliar
devices degrade instead of crashing. See ggmlFATuningParams().

The path and tuning are selected per (shape, device) in prepare(); a tuning
change re-requests the pipeline from the pool. Every path leaves the KV
cache in the identical state (fused append), so the graph meaning does not
depend on which one runs.

The mask's span bounds a query to its last span keys, a sliding window
(e.g. Gemma3's local layers). A ring cache (kvCacheRing) then keeps only the
last kvCacheSize positions, position p at row p % kvCacheSize; the counter
keeps counting positions. A dispatch overwrites the rows of its Tkv oldest
positions, so kvCacheSize >= span + Tkv - 1 keeps every query's span whole.
The node rounds a ring up to kvRingAlign, so no key tile wraps it.

kvCacheSize = 0 is orthogonal to both: it drops the cache itself, so a
dispatch attends its own tokens and nothing else (an encoder, e.g. SigLIP).
Nothing is allocated and no counter is wired — the cache and counter
descriptors go null, and the 0 a null counter reads back is the whole trick.

Tkv may differ from Tq: queries and keys align at their ends, so query t
reaches the keys up to its own mask end shifted by Tkv - Tq (e.g. a decoder
over its encoder's output, Tq queries on Tkv keys).
With a cache, K / V may stay unwired (Tkv = 0): the dispatch attends the
cache alone and appends nothing, and the model keeps the counter at the
rows held.

No mask (setCausalMask never called) binds the mask null: every query
attends every key it reaches.

Split-K (decode): when the grid would underfill the GPU, the cache sweep is
split into kCount segments accumulating partials (partialO/partialML) that
a second pass (attention-fa-reduce) folds with the online-
softmax merge. kCount is the dispatch grid, so it holds for the whole
recording and is derived from occupancy alone; where the segments fall
follows the cache, which grows under a recording, so the kernel derives that
from kvCacheCount per forward.

**커널**

- `vai/shaders/attention-fa-intel.comp`
- `vai/shaders/attention-fa-cm.comp`
- `vai/shaders/attention-fa.comp`
- `vai/shaders/attention-fa-reduce.comp`
- `vai/shaders/attention.comp`

---

## Concat

join inputs along one axis

소스: [`vai/nodes/concat.h`](../../vai/nodes/concat.h)

**시그니처**

- 입력: in0 .. inN [same shape except along the concat axis]
- 출력: out0 the inputs joined along axis, in input order

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0, in1, …` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `axis` | `int` | `0` | negative counts from the end |
| `inputCount` | `uint32_t` | `2` |  |
| `allowPartialInputs` | `bool` | `false` |  |

**동작 및 제약**

All inputs must have identical dtypes, ranks, and dimensions except along axis. A
negative axis counts from the end.

What the axes ahead of the cut multiply to decides the cost, not the axis number. At
1 each input is one contiguous run of the output, so buffer copies place them and no
shader runs — axis 0 is always this case, and so is a deeper axis with nothing but
size-1 axes in front. One input arriving is then the whole output and out0 aliases it.
Past 1 the runs interleave and a gather kernel reads them, which is where the 8-input
limit and the dtype conversion live.

allowPartialInputs lets a recording bring only some of them; the rest are not joined.
Off, every input is required, so a recording that cannot supply one drops this node.

**커널**

- `vai/shaders/concat.comp`

---

## Convolution

2D convolution with device-selected direct, implicit, or Winograd plans

소스: [`vai/nodes/convolution.h`](../../vai/nodes/convolution.h)

**시그니처**

- 입력: in0 [..., Ci, Hi, Wi], optional residual [..., Co, Ho, Wo]
- 파라미터: weight [Co, Ci, K, K], optional scale [Co] and bias [Co]
- 출력: out0 [..., Co, ceil(Hi / stride), ceil(Wi / stride)]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input |  |
| `out0` | output |  |
| `bias` | persistent |  |
| `scale` | persistent |  |
| `weight` | persistent |  |
| `split` | temporary |  |
| `winM` | temporary |  |
| `winV` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `inChannels` | `uint32_t` | — |  |
| `outChannels` | `uint32_t` | — |  |
| `kernelSize` | `uint32_t` | — |  |
| `stride` | `uint32_t` | — |  |
| `acceptResidual` | `bool` | `false` | input "residual" |
| `residualOrder` | `vai::ResidualOrder` | `vai::ResidualOrder::AfterEpilogue` |  |
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |

**동작 및 제약**

The "scale" and "bias" slots always exist; each applies only when the weight
loader filled it. Odd kernels use symmetric same-padding, so each spatial
output size is ceil(inputSize / stride).
acceptResidual opens the "residual" input (input dtype), added after the epilogue
or, with ResidualOrder::BeforeEpilogue, before it.
The axes ahead of Ci are a batch of images. The plan is one image's; every
dispatch stacks the images on its z axis, and each kernel offsets its buffers
to its image.

**커널**

- `vai/shaders/conv-k1s1-gemm-cm.comp`
- `vai/shaders/conv-k1s1-gemm.comp`
- `vai/shaders/conv-k3s2-gemm-cm.comp`
- `vai/shaders/conv-k3s1-implicit-gemm-cm.comp`
- `vai/shaders/conv-k3s1-implicit-gemm.comp`
- `vai/shaders/conv-implicit-gemm.comp`
- `vai/shaders/conv-implicit-gemm-cm.comp`
- `vai/shaders/conv-k3s2-gemm.comp`
- `vai/shaders/cm-transform.comp`
- `vai/shaders/conv-k3s1-winograd-filter-transform.comp`
- `vai/shaders/conv-k3s1-winograd-input.comp`
- `vai/shaders/conv-k3s1-winograd-output.comp`
- `vai/shaders/bn-reduce.comp`
- `vai/shaders/conv-k3s1-winograd-gemm.comp`
- `vai/shaders/conv-k3s1-winograd-gemm-cm.comp`

---

## CyclicTableReader

read successive table rows, wrapping at the end

소스: [`vai/nodes/cyclic-table-reader.h`](../../vai/nodes/cyclic-table-reader.h)

**시그니처**

- 입력: in0 [*] only scheduling: the node runs when it is fed
- 파라미터: table [numRows, ...] rows read in cyclic order, at the loader's dtype
- 출력: out0 table's shape without the leading axis, at oDtype

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `table` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `numRows` | `uint32_t` | — | number of rows in the table |

**동작 및 제약**

Run k after rewind() copies row k % numRows, so a loop whose per-step
values depend on the step alone (e.g. pi0.5's adaRMS modulation) reads them
with no host write, the way TimeEmbedding walks t. rewind() starts over.

**커널**

- `vai/shaders/cyclic-table-reader.comp`

---

## EleAffine

element-wise affine, O = scale * in0 + shift

소스: [`vai/nodes/ele-affine.h`](../../vai/nodes/ele-affine.h)

**시그니처**

- 입력: in0 [..., D] scale broadcast to in0's shape (right-aligned, size-1 axes broadcast); scale 1 when absent shift likewise; shift 0 when absent
- 출력: out0 in0's shape

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `scaleAsInput` | `bool` | `false` | input instead of persistent |
| `shiftAsInput` | `bool` | `false` |  |

**동작 및 제약**

scale / shift are inputs under scaleAsInput / shiftAsInput, otherwise
parameters at the loader's dtype. Each applies when present: filled by the
loader, or wired; at least one has to be. Each may broadcast over any axes as long as the axes it keeps
are adjacent, e.g. [1, D] over [T, D], or [C, 1, 1] over [C, H, W].

An add is shift alone, a multiply scale alone, and a gated residual x + g * y
both, EleAffine(in0 = y, scale = g, shift = x) (e.g. pi0.5's adaRMS layers).

**커널**

- `vai/shaders/ele-affine.comp`

---

## Flatten

Flatten tensor to 1D

소스: [`vai/nodes/flatten.h`](../../vai/nodes/flatten.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [N] where N = total number of elements (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

elements is the input's total element count. The output aliases the input buffer.

---

## Gate

conditional branch (no computation)

소스: [`vai/nodes/gate.h`](../../vai/nodes/gate.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: then [same shape], else [same shape] — both alias in0's buffer

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `else` | output |  |
| `then` | output |  |

**동작 및 제약**

One branch is recorded per command buffer. The engine drops every node that
only the closed branch reaches, so those nodes neither prepare() nor run();
a node both branches reach (the merge point and everything past it) is kept.
Leaving "else" unwired makes a closed gate drop the whole downstream.

The condition is host state, so flipping it re-records. Weights are uploaded
by uploadModelParams(), which ignores gates — a branch closed on the first forward
still gets its weights on the device.

Two producers may share one input slot only where a gate's branches merge;
sortNodes() rejects any other fan-in. The branches need not agree on shape or
dtype: a flip re-records, so every recording derives the merge point and its
downstream from whichever branch survived it.

---

## GatedMul

element-wise multiply with one operand activated (GLU gating)

소스: [`vai/nodes/gated-mul.h`](../../vai/nodes/gated-mul.h)

**시그니처**

- 입력: in0 [..., 2H]
- 출력: out0 [..., H] = activation(in0[..., j]) * in0[..., H + j]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `activation` | `vai::Epilogue` | `vai::Epilogue::Silu` |  |

**동작 및 제약**

The two halves arrive in ONE buffer: the projections feeding them are a
single Linear over the concatenated weights, so the leading half is whatever
that concatenation put first. It is the activated one.

activation names the GLU variant — silu: SwiGLU, gelu: GEGLU, relu: ReGLU,
none: bilinear.

**커널**

- `vai/shaders/gated-mul.comp`

---

## GroupNorm

group normalization over a CHW feature map

소스: [`vai/nodes/group-norm.h`](../../vai/nodes/group-norm.h)

**시그니처**

- 입력: in0 [..., C, H, W], optional residual [..., C, H, W]
- 출력: epilogue(scale[c] * (in0 - μ[g(c)]) * rsqrt(σ²[g(c)] + eps) + shift[c]) + residual

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `groups` | `uint32_t` | — |  |
| `eps` | `float` | `1e-5f` |  |
| `coeffsAsInputs` | `bool` | `false` | scale / shift are inputs, not parameters |
| `acceptResidual` | `bool` | `false` | input "residual" |
| `residualOrder` | `vai::ResidualOrder` | `vai::ResidualOrder::AfterEpilogue` |  |
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |

**동작 및 제약**

The channel axis is third from the end; the axes before it are independent
samples. scale / shift [C] are parameters, or inputs under coeffsAsInputs
(e.g. AdaGN's timestep modulation); shift applies when present, filled by
the loader or wired. acceptResidual opens the "residual" input (input
dtype), added after the epilogue or, with ResidualOrder::BeforeEpilogue,
before it. The channels split into `groups` consecutive groups; each group's statistics
are computed at run time over its C/groups channels and all of their spatial
positions (unlike BatchNorm, which reads μ/σ² trained offline). One
workgroup per group (group-norm), statistics from a shifted 1-pass
reduction accumulated in fp32 regardless of I/O dtypes — the reduction is
subgroupAdd two-level when the device has subgroup arithmetic, a
shared-memory tree otherwise.

Both the workgroup size and the shape itself are baked into the pipeline, so
it is requested in prepare() once the shape is known and re-requested when
any of that changes (a no-op while it stays put) — same arrangement as
LayerNorm. The shape is in there because the kernel's per-element
channel index is j / HW: a push-constant HW costs an integer division per
element, a specialization-constant one is a shift or a multiply-high. That
makes the pipeline key per (dtype, group geometry), which is a handful of
entries for a model — a ResNet-18 stage or a UNet1D level each contribute
one, and every block inside it reuses that.

**커널**

- `vai/shaders/group-norm.comp`

---

## Identity

Pass-through node (no computation)

소스: [`vai/nodes/identity.h`](../../vai/nodes/identity.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [same shape] (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

The output aliases the input buffer; no computation or copy occurs.

---

## IndexCopy

write source slices at listed indices, as torch.index_copy

소스: [`vai/nodes/index-copy.h`](../../vai/nodes/index-copy.h)

**시그니처**

- 입력: in0 [d0, .., d(r-1)] in1 in0's shape with d(axis) replaced by N
- 출력: out0 in0 with out0[.., index[n], ..] = in1[.., n, ..] along axis

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `axis` | `uint32_t` | `0` |  |

**동작 및 제약**

The index is N uint32 in a host-visible buffer the model owns and hands over
with setIndex. It is read at dispatch time, so rewriting it after sync() needs
no re-recording. An index past d(axis) leaves its slice unwritten.

out0 aliases in0 and only the N slices are written (= index_copy_), so no
other node may read in0 after this one. in1 may be left out of a recording,
and out0 then aliases in0 with nothing recorded. Bytes move as they are, so
in0 and in1 share a dtype.

**커널**

- `vai/shaders/index-copy.comp`

---

## LayerNorm

layer normalization over the feature axis, single kernel

소스: [`vai/nodes/layer-norm.h`](../../vai/nodes/layer-norm.h)

**시그니처**

- 입력: activations [..., D]
- 출력: normalized [..., D]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `eps` | `float` | `1e-5f` |  |
| `coeffsAsInputs` | `bool` | `false` | scale / shift are inputs, not parameters |
| `acceptResidual` | `bool` | `false` | input "residual", added to the output |

**동작 및 제약**

scale / shift   [D] (or [1, D] as an input); parameters, or inputs computed
upstream under Config::coeffsAsInputs (e.g. DiT's adaLN).
shift applies when present: filled by the loader, or wired.

O[i, j] = (I[i, j] - mean(I[i, :])) * rsqrt(var(I[i, :]) + eps) * scale[j] + shift[j]

One workgroup per row (layer-norm); the row statistics come from a
shifted 1-pass reduction (Chan–Golub–LeVeque) accumulated in fp32
regardless of I/O dtypes. Two spec-selected variants share the kernel:

- reduction: subgroupAdd two-level when the device has subgroup arithmetic
(subgroup size pinned to max — Intel varies it otherwise), shared-memory
tree fallback everywhere else.
- loads: vec4 when I and O are fp32 and featSize % 4 == 0, scalar otherwise.

Geometry / reduction / vec4 are selected per (featSize, device) in
lnTuning(); the workgroup size depends on featSize, so the pipeline is
requested in prepare() once the shape is known, and a tuning change
re-requests it from the pool (no-op while the tuning stays put).

**커널**

- `vai/shaders/layer-norm.comp`

---

## Linear

O = I · Wᵀ, split between two output decompositions

소스: [`vai/nodes/linear.h`](../../vai/nodes/linear.h)

**시그니처**

- 입력: [..., inDim]
- 출력: [..., outDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input | 옵션에 따라 존재 |
| `out0` | output |  |
| `bias` | persistent |  |
| `weight` | persistent |  |
| `partial` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `inDim` | `uint32_t` | — |  |
| `outDim` | `uint32_t` | — |  |
| `acceptResidual` | `bool` | `false` | input "residual" |
| `residualOrder` | `vai::ResidualOrder` | `vai::ResidualOrder::AfterEpilogue` |  |
| `epilogue` | `vai::Epilogue` | `vai::Epilogue::None` |  |
| `scalarScale` | `float` | `1.0f` | s, right after the epilogue |
| `splitSize` | `uint32_t` | `0` | split-K: K length per split (0: the vendor's count per shape) |

**동작 및 제약**

vai naming: O [M, N], I [M, K] (M = flattened batch), W [N, K]
(N = outDim, K = inDim).

- slab (M <= 8, linear-slab): a workgroup owns wgN whole columns of O,
so K is split across its threads and reduced; bias / epilogue are fused
in-kernel. See useSlabPath().
- tile (M > 8): a workgroup owns a wgM x wgN tile. Two kernels share the
frame, chosen per device in selectLinearTile(): scalar
(linear-tile, each thread accumulates its own outputs) and
cooperative-matrix (linear-cm, subgroup MMA tiles; NVIDIA's Fill tiles
run linear-cm-nv, which masks the K tail per k-block).
- Intel (useIntelTile; every M, slab included): linear-tile-intel, a port of
OpenVINO's fully_connected_gpu_bf_tiled, reading W packed once on the GPU
(packWeightIntel); its tile choice is intelTileParams(), its split
intelTileSplitCount().
Everything below applies to every tile kernel:

The tile params are selected per output shape in prepare(); when a shape
change selects a different tile, the pipeline is re-requested from the pool.

Split-K (splitSize option; 0 selects the vendor's rule per shape)
accumulates partials P[splitCount, M, N] (oDtype) in a temporary buffer
and folds them in a second pass (linear-reduce); bias / epilogue / the
residual live in that reduce pass, so requesting them without splitting still
runs the reduce as a plain epilogue pass (splitCount = 1).

O = s · epilogue(I · Wᵀ + bias) + R      residualOrder AfterEpilogue
O = s · epilogue(I · Wᵀ + bias + R)      residualOrder BeforeEpilogue

s is Config::scalarScale (e.g. dt of an Euler step x + dt · v), 1 by default.

"bias" (its own dtype) is a parameter and "residual" R (oDtype) an input. Each is
[M, N] or rows that repeat over M: [N], or one image's position table under
a batch. The "bias" slot always exists; the bias applies only when the
weight loader filled it.

**커널**

- `vai/shaders/linear-pack-intel.comp`
- `vai/shaders/linear-slab.comp`
- `vai/shaders/linear-tile-intel.comp`
- `vai/shaders/linear-cm-nv.comp`
- `vai/shaders/linear-cm.comp`
- `vai/shaders/linear-tile.comp`
- `vai/shaders/linear-reduce.comp`

---

## Mul

Element-wise multiplication

소스: [`vai/nodes/mul.h`](../../vai/nodes/mul.h)

**시그니처**

- 입력: in0 [any shape], in1 [same shape]
- 출력: out0 [same shape] = in0 * in1

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**동작 및 제약**

Inputs must have identical shapes and dtypes.

**커널**

- `vai/shaders/mul.comp`

---

## NearestUpsample

Replicates CHW pixels by an integer scale factor.

소스: [`vai/nodes/nearest-upsample.h`](../../vai/nodes/nearest-upsample.h)

**시그니처**

- 입력: in0 [C, H, W]
- 출력: out0 [C, H*scale, W*scale]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

scaleFactor must be greater than one. Each input pixel is replicated without
interpolation.

**커널**

- `vai/shaders/upsample.comp`

---

## Permute

Permutes the axes of a tensor of any rank.

소스: [`vai/nodes/permute.h`](../../vai/nodes/permute.h)

**시그니처**

- 입력: in0 [D0, .., Dr-1], r = perms.size()
- 출력: axes reordered by perms; reshape-only fast path when layout is unchanged

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

Output dimension i is input dimension perms[i], and perms must be a permutation of
{0, .., r-1} with r at most PERMUTE_MAX_RANK. When the permutation preserves memory
order and the dtype is unchanged, the output aliases the input; otherwise it uses new
contiguous storage.

The rank is a specialization constant, so the pool holds one pipeline per rank.

**커널**

- `vai/shaders/permute.comp`

---

## Pool2d

2D pooling with a square kernel (MaxPool2d, AvgPool2d)

소스: [`vai/nodes/pool2d.h`](../../vai/nodes/pool2d.h)

**시그니처**

- 입력: in0 [..., C, H, W], or [..., H, W, C] with channelsLast
- 출력: [..., C, Ho, Wo], or [..., Ho, Wo, C] where Ho = (H + 2*padding - kernelSize) / stride + 1, rounded up with ceilMode

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `kernelSize` | `uint32_t` | — |  |
| `stride` | `uint32_t` | — |  |
| `padding` | `uint32_t` | `0` |  |
| `ceilMode` | `bool` | `false` | a last window that would start in the right padding is dropped |
| `countIncludePad` | `bool` | `true` | Avg: padding counts in the divisor, as PyTorch's default |
| `channelsLast` | `bool` | `false` |  |

**동작 및 제약**

channelsLast reads a token grid as it is laid out (rows of embedDim, e.g. a
ViT's patches viewed as [G, G, D]), so no transpose is needed around the pool.

**커널**

- `vai/shaders/pool2d.comp`

---

## Reshape

Reshape tensor to specified dimensions

소스: [`vai/nodes/reshape.h`](../../vai/nodes/reshape.h)

**시그니처**

- 입력: in0 [any shape]
- 출력: out0 [dims] (shares buffer with input)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**동작 및 제약**

A dim is a size, 0 for the input's dimension at that same position, or -1 for
whatever the element count leaves over, at most once (= ONNX). 0 carries a size only
the caller's data knows without the graph naming it. The resolved dims must multiply
to the input's element count. The output aliases the input buffer.

---

## RmsNorm

RMS normalization over the feature axis, single kernel

소스: [`vai/nodes/rms-norm.h`](../../vai/nodes/rms-norm.h)

**시그니처**

- 입력: in0 activations [..., D]
- 출력: out0 normalized [..., D]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `residual` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `eps` | `float` | `1e-5f` |  |
| `coeffsAsInputs` | `bool` | `false` | scale / shift are inputs, not parameters |
| `acceptResidual` | `bool` | `false` | input "residual", added to the output |

**동작 및 제약**

scale / shift   [D] (or [1, D] as an input), each applied when present: a
parameter the loader filled, or an input wired upstream under
Config::coeffsAsInputs (e.g. pi0.5's adaRMS)
residual       [..., D] as in0, an input under Config::acceptResidual

O[i, j] = I[i, j] * rsqrt(mean(I[i, :]^2) + eps) (* scale[j]) (+ shift[j]) (+ residual[i, j])

The row statistic is a 1-pass sum of squares accumulated in fp32 regardless
of I/O dtypes. The workgroup size depends on D, so the pipeline is requested in
prepare() once the shape is known.

**커널**

- `vai/shaders/rms-norm.comp`

---

## Rope

rotary position embedding (half-split rotation)

소스: [`vai/nodes/rope.h`](../../vai/nodes/rope.h)

**시그니처**

- 입력: in0 [.., H, Tq, Dh] (Q or K, Dh even)
- 파라미터: freqs [Dh/2] per-dimension angular frequency, built by the constructor
- 출력: out0 [.., H, Tq, Dh]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `freqs` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `cfg` | `RopeConfig` | — | frequencies; describe the whole model |
| `headDim` | `uint32_t` | — | Dh, must be even |

**동작 및 제약**

The rotation depends on the token position alone, so anything ahead of the head
axis is one more sweep and folds into the head count.

Positions continue from setKvCacheCounter()'s buffer, so a KV-cache decode
step rotates by the absolute position rather than by its offset in this forward.

cfg.type picks how the frequencies are scaled: Default leaves them alone,
Linear divides every one by factor, and Llama3 divides only the long
wavelengths, interpolating the band between the two frequency bounds.

**커널**

- `vai/shaders/rope.comp`

---

## RowSliceScatter

write slices into rows, as torch.slice_scatter on dim 0

소스: [`vai/nodes/row-slice-scatter.h`](../../vai/nodes/row-slice-scatter.h)

**시그니처**

- 입력: in0 [T, D] in1 [B, S, D] or [S, D] (S == sliceSize; B slices, one when rank 2)
- 출력: out0 [T, D], in0 with rows [rowStarts[b], rowStarts[b] + S) taken from slice b

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `in1` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `sliceSize` | `uint32_t` | — | S: rows in one slice; 0 when in1 is never fed |

**동작 및 제약**

in1 may be left out of a recording, and out0 then aliases in0 with nothing recorded.
Rows move as bytes, so in0 and in1 share a dtype. The row starts are host state
that the recorded copies bake in: setRowStarts re-records only when they change.

---

## Slice

Take a run of consecutive indices along one axis

소스: [`vai/nodes/slice.h`](../../vai/nodes/slice.h)

**시그니처**

- 입력: in0 [d0, .., d(r-1)] (rank >= 2)
- 출력: out0 the same shape with d(axis) replaced by count

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `axis` | `uint32_t` | `0` | the axis to cut; 0 takes a contiguous block |
| `start` | `int64_t` | — | first index on the axis; negative counts from the end |
| `count` | `uint32_t` | `1` | how many consecutive indices to take |

**동작 및 제약**

A negative start counts from the end, with -1 selecting the last index.

What the axes ahead of the cut multiply to decides the cost, not the axis number.
At 1 the run is one contiguous block, so out0 aliases it when the offset satisfies
device alignment and is one buffer copy otherwise — axis 0 is always this case, and
so is a deeper axis with nothing but size-1 axes in front. Past 1 the run is strided,
count indices every d(axis), and a gather kernel reads it out. The shape decides the
path; the caller writes the same node either way.

**커널**

- `vai/shaders/slice.comp`

---

## TimeEmbedding

sinusoidal embedding of a timestep the node counts down itself

소스: [`vai/nodes/time-embedding.h`](../../vai/nodes/time-embedding.h)

**시그니처**

- 입력: in0 [.., rows, *] only its shape is read: one embedding per row
- 출력: out0 [.., rows, embedDim]

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `embedDim` | `uint32_t` | — | even; sin in the first half, cos in the second |
| `minPeriod` | `float` | — | the fastest sinusoid, in units of t |
| `maxPeriod` | `float` | — | ... and the slowest |
| `numSteps` | `uint32_t` | — | runs before t is back at 1 |

**동작 및 제약**

out0[r, i]        = sin(2*pi * t / p_i)
out0[r, half + i] = cos(2*pi * t / p_i),   p_i = minPeriod * (maxPeriod / minPeriod)^(i / (half - 1))

Run k after resetTime() embeds t = 1 - (k % numSteps) / numSteps, so a
flow-matching loop of numSteps steps walks t from 1 down with no host write.
resetTime() starts it over (e.g. pi0 after a prefill); the count wraps after
2^32 workgroups, so a caller that never resets does so within that.

**커널**

- `vai/shaders/time-embedding.comp`

---

## TokenEmbedding

token embedding, optionally fused with positional embedding

소스: [`vai/nodes/token-embedding.h`](../../vai/nodes/token-embedding.h)

**시그니처**

- 입력: in0 token IDs [..., T] (uint32)
- 출력: out0 embedding [..., T, D] (fp32)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `posWeight` | persistent |  |
| `weight` | persistent |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `vocabSize` | `uint32_t` | — |  |
| `embedDim` | `uint32_t` | — |  |
| `posTableSize` | `uint32_t` | `0` | learned positional table rows; 0 = token only |
| `scale` | `float` | `1.0f` | multiplies the looked-up embedding |

**동작 및 제약**

Leading axes are more sequences swept in one dispatch, the way a Linear takes its
leading axes as more rows, so the output keeps the input's rank.

posTableSize == 0 : O[n] = scale * W[ids[n]]                       (e.g. Llama3 — RoPE)
posTableSize  > 0 : O[n] = scale * W[ids[n]] + posW[cachedCount + n]  (e.g. GPT-2)

Only Gemma uses scale (sqrt(D)). It stays out of the weights because a tied LM head
reads the same table. posTableSize > 0 adds a "posWeight" [posTableSize, D] slot
and needs setKvCacheCounter() to supply cachedCount; positions restart at each
sequence, which the kernel only has to work out when there is more than one.

**커널**

- `vai/shaders/token-embedding.comp`

---

## TopKSampling

row-wise top-k sampling, one kernel run as two passes

소스: [`vai/nodes/topk-sampling.h`](../../vai/nodes/topk-sampling.h)

**시그니처**

- 입력: logits [.., N]
- 출력: out0 [..] (one raw uint32 sampled index per row, in 4-byte slots)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |
| `cands` | temporary |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `k` | `uint32_t` | `40` |  |
| `temperature` | `float` | `1.0f` |  |

**동작 및 제약**

Every axis ahead of N is a row, the way a Linear takes its leading axes, and they
come back on the output where they went in.

Keeps the k largest logits of each row, applies a temperature softmax over
them, and samples by inverse CDF with a host-provided uniform refreshed
every forward (setUniform()).

Two dispatches of one kernel: G workgroups each select the top-k of one
ceil(N/G)-logit slice — exact while the slice stays register-resident — then
one workgroup merges the G*k candidates and samples. Small rows (G == 1)
skip the second dispatch.

Geometry (workgroup size, register list length), the reduction variant and G
are selected per (shape, k, device) in topkTuning(); the workgroup size is a
spec constant, so the pipeline is requested in prepare() once the shape is
known, and a tuning change re-requests it from the pool (no-op while the
tuning stays put).

**커널**

- `vai/shaders/topk-sampling.comp`

---

## TypeCast

Casts each tensor element to another dtype.

소스: [`vai/nodes/type-cast.h`](../../vai/nodes/type-cast.h)

**시그니처**

- 입력: in0 [any shape], any dtype
- 출력: out0 [same shape] (oDtype)

**슬롯**

| 슬롯 | 종류 | 비고 |
|---|---|---|
| `in0` | input |  |
| `out0` | output |  |

**Config**

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `oDtype` | `vai::Dtype` | — | required: the dtype to cast to |

**동작 및 제약**

in0 already at oDtype makes out0 the input tensor itself: no pipeline, no
dispatch.

**커널**

- `vai/shaders/cast.comp`

---
