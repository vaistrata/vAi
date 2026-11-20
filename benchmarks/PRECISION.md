
# pi0.5 Benchmark Precision Configuration

All comparison frameworks primarily use FP16 model weights, but some parameters remain in FP32. They also differ in activation storage dtypes, GEMM and attention accumulation precision, and residual dtypes.

This document describes the computation precision used by each framework in the [pi0.5 benchmarks](pi05/README.md) and explains how it was verified.

The precision analysis distinguishes three categories:

- **Configured precision:** Dtypes and computation precision explicitly specified in the source code or runtime configuration.
- **Graph precision:** Tensor dtypes identified in exported ONNX graphs or TensorRT engines.
- **Execution precision:** Computation precision verified through GPU kernel instructions and runtime execution information.

## Summary

| Component | vAi | vla.cpp | TensorRT Edge-LLM | MIGraphX |
|---|---|---|---|---|
| Model weights | FP16 (normalization parameters and condition tables are FP32) | FP16 (patch conv, action input/output projections, time MLP, and normalization parameters are FP32) | FP16 | FP16 (only RoPE frequency buffers are FP32) |
| Activation storage | FP16 block branches / residuals: FP32 for vision and language, FP16 for action | FP32 (converted to FP16 immediately before weight GEMM) | FP16 (FP32 for norm, AdaRMS, and action residual sections) | FP16 (images, timestep, and x are FP32) |
| K/V cache | FP16 | FP32 | FP16 | FP16 |
| Vision weight GEMM accumulation | FP32 | **FP16** | **FP16** | Determined at compile time (not verified) |
| Language / action weight GEMM accumulation | NVIDIA: **FP16** / AMD: FP32 | **FP16** | **FP16** | Determined at compile time (not verified) |
| Attention QKᵀ / PV accumulation | FP32 | TF32 multiplication, FP32 accumulation | Vision, language: **FP16** / action: FP32 | Determined at compile time (not verified) |
| Softmax and normalization (RMSNorm, LayerNorm, AdaRMS) | FP32 for statistics and softmax, FP16 output | FP32 | FP32 | FP32 for softmax and RMSNorm variance / FP16 otherwise (based on ONNX) |
| Vision / language residuals | FP32 | FP32 | FP16 | FP16 |
| Action expert residuals | FP16 | FP32 | FP32 | FP16 |
| Final action update (Euler integration) | FP32 | FP32 | Not applicable | FP32 |

> [!NOTE]
> This table summarizes the precision identified from source code, exported graphs, or GPU kernels for the analyzed execution configurations.
>
> Actual computation precision may vary depending on the GPU architecture, framework version, compiler kernel selection, backend library settings, and kernel fusion.
>
> **FP16/FP32 values represent the precision identified in the analyzed configurations. They do not imply that the same precision applies to every GPU or execution environment.**
>
> `Determined at compile time (not verified)` means that the actual accumulation precision is determined during compilation, but the executed kernels have not been verified. `Not applicable` means that the corresponding operation does not apply.
>
> Tensor dtypes in an ONNX graph alone are not sufficient to determine the actual accumulation precision of GPU kernels.

## Verification Methods

### vla.cpp

The pi0.5 inference path and the ggml-cuda backend source code of [vla.cpp](https://github.com/VinRobotics/vla.cpp) were analyzed.

The analysis covered runtime options applied to the pi0.5 model, dtype conversions for weights and activations, and CUDA GEMM call paths.

The following observations were made:

- The pi0.5 inference path uses the `--weight-dtype` option.
- In the analyzed pi0.5 path, the `--act-dtype` and `--flash-attn` options do not affect the computation path.
- Activations are maintained in FP32 and converted to FP16 immediately before FP16 weight GEMM operations.
- The analyzed FP16 weight GEMM path uses the cuBLAS `CUBLAS_COMPUTE_16F` compute type.
- The attention matrix multiplication path was identified as using TF32 multiplication with FP32 accumulation.

`CUBLAS_COMPUTE_16F` specifies FP16 compute mode. Therefore, this document classifies the GEMM accumulation precision of that call path as FP16.

However, the compute type specified through the cuBLAS API must be distinguished from the precision of every instruction actually executed on the GPU. Further analysis of the executed CUDA kernels is required to determine the internal computation precision of individual kernels.

References:

- [vla.cpp GitHub](https://github.com/VinRobotics/vla.cpp)
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

### TensorRT Edge-LLM

The pi0.5 inference path of [NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) was analyzed using the model graph, built TensorRT engine, and actual GPU kernels.

Precision was verified through the following steps:

1. Examined tensor dtypes in the ONNX model exported using the official tools.
2. Inspected the layer implementations and selected kernels in `layers.json` from the built TensorRT engine.
3. Collected GPU kernels loaded during inference using CUPTI.
4. Analyzed SASS matrix multiplication instructions (`HMMA.*.F16`, `HMMA.*.F32`) to identify computation precision.
5. Used `nsys` to identify the kernels actually executed and cross-checked them against the analysis results.

This process distinguished the dtypes defined in the model graph from the computation precision of the GPU kernels actually executed.

Because the interpretation of SASS instructions may depend on the GPU architecture and instruction format, the findings are limited to the analyzed GPU and TensorRT engine configuration.

References:

- [NVIDIA TensorRT Edge-LLM GitHub](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- [NVIDIA CUPTI Documentation](https://docs.nvidia.com/cupti/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

### MIGraphX

The precision of [MIGraphX](https://github.com/ROCm/AMDMIGraphX) was analyzed using [export-onnx.py](pi05/migraphx/export-onnx.py), which generates the ONNX bundle used by the benchmark.

This script converts model parameters from the FP32 checkpoint to FP16 before exporting the ONNX models.

During export, it also checks the graph input/output boundaries and parameter dtypes.

The following properties were identified:

- Dtypes of model weights and selected buffers
- Dtypes of graph input and output tensors
- Dtype conversions between operations
- FP16 and FP32 computation paths explicitly represented in the ONNX graph

However, tensor dtypes specified in an ONNX graph alone are not sufficient to determine the actual accumulation precision of GPU kernels.

The accumulation behavior depends on the kernels selected by MIGraphX during compilation and their backend implementations.

Therefore, this document does not verify the actual GEMM or attention accumulation precision of MIGraphX. These entries are marked as `Determined at compile time (not verified)` in the summary table.

References:

- [AMD MIGraphX GitHub](https://github.com/ROCm/AMDMIGraphX)

### vAi

vAi allows the output dtype and accumulation precision of individual nodes to be configured directly during model construction.

Node dtypes are assigned per tower in each network constructor. The relevant implementation is defined in [net.h](../models/experimental/pi05/net.h).

Nodes without explicitly assigned dtypes follow the defaults defined by the network or node.

A node's input dtype is generally determined by the output dtype of the preceding connected node.

Precision is configured in three areas:

| Configuration target | Configuration location | Default |
|---|---|---|
| Weights and node outputs | `wDefaultDtype`, `oDefaultDtype` in `NeuralNetConfig` | FP16, FP32 |
| Per-node output and accumulation | Node methods `setODtype`, `setAccDtype` (called in the network constructor) | Output: network default. Linear accumulation: FP32 on AMD, weight dtype otherwise |
| K/V cache | `RuntimeConfig::kvCacheDtype` | FP16 |

Node dtype setters require a `DtypeKey` argument.

This key is obtained through `NeuralNet::dtypeKey()` and is used to restrict precision configuration to the model network implementation.

Model-specific precision settings are applied in the network constructor.

For example, [PR #46](https://github.com/vaistrata/vAi/pull/46) configures the precision of selected operations in the vision blocks as follows:

```cpp
const auto f16 = vai::Dtype::Fp16, f32 = vai::Dtype::Fp32;
const auto key = dtypeKey();

for (auto& b : visionEncoder.blocks)
{
    auto& a = b.mhAttn;
    a.projQ.setAccDtype(f32, key);  // Accumulation: FP32
    b.ffn.fc1.setAccDtype(f32, key);
    b.norm1.setODtype(f16, key);    // Output: FP16
    a.attn.setODtype(f16, key);
    // ...
}
```

These settings specify the output dtype and accumulation precision of individual nodes.

However, not all operation precision can be modified through dtype setters. Some operations have their precision determined by the GPU backend kernel implementation.

#### Attention

The Flash Attention kernels on NVIDIA and AMD accumulate QKᵀ and P·V matrix multiplications in FP32 and compute softmax in FP32.

The relevant implementation is defined in [attention.h](../vai/nodes/attention.h).

`setQkAccDtype` and `setOAccDtype` apply only to Intel kernels. Therefore, these setters do not change the Flash Attention accumulation precision on NVIDIA or AMD.

#### Normalization

RMSNorm and LayerNorm accumulate row statistics in FP32 regardless of the input and output dtypes.

Relevant implementations:

- [rms-norm.h](../vai/nodes/rms-norm.h)
- [layer-norm.h](../vai/nodes/layer-norm.h)

Therefore, even when the input activations are FP16, the statistics required for normalization can be computed in FP32.

## Considerations When Comparing Precision

When comparing benchmark precision, weight dtypes, activation dtypes, and accumulation precision must be distinguished.

Even when frameworks use the same FP16 weights, their numerical results and execution times may differ due to the following factors:

- GEMM input conversion and accumulation precision
- Precision of attention QKᵀ and P·V operations
- Precision of softmax and normalization statistics
- Dtypes maintained across residual connections
- Compiler operation fusion and GPU kernel selection
- GPU architecture and backend library versions

Higher computation precision also does not always guarantee lower prediction error. Actual accuracy depends on operation order, dtype conversions, kernel implementations, and the propagation of numerical errors.

The precision values in this document apply to the analyzed implementations and execution configurations. Actual computation precision must be verified again when the framework version or GPU changes.
