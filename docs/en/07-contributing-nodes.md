# Writing custom nodes and kernels

A node generally represents a single operation in a neural network and corresponds to a single unit of execution from the GPU's perspective.

Each node executes the GPU kernel that performs the operation and is responsible for binding the arguments and buffers required for kernel execution and controlling the actual execution. Put simply, a node can be viewed as a unit that handles one operation and the control required to execute that operation on the GPU.

If an operation that is not supported by default is needed, it can be implemented directly by referring to this section. This section explains the process using the simple `activation.comp` and `activation.h` as examples.

```
vai/nodes/vai-<name>.h        node (header only)
vai/shaders/vai-<name>.comp   kernel
```

A node used only by a specific model may be placed in the model folder.
[`models/internal/yolov8/nodes/`](../../models/internal/yolov8/nodes/) is an example.

## Code conventions

**Consistency with the conventions is the top priority**. Follow the existing naming, patterns, and helpers, and do not create exceptions. However, kernels only need to follow the header stamp format, and the remaining code conventions are not enforced.

### Header stamp

Both nodes and kernels begin and end with `/* ... */` block comments and `=` fences. The node stamp is the original template used to generate the [provided nodes](06-nodes.md), so its format must be preserved.

```cpp
/*=============================================================================
  RmsNorm - RMS normalization (normalize over last dimension without mean centering)
  Input : in0 [..., D]
  Params: scale [D]
  Output: out0 [..., D] = in0 / rms(in0) * scale

=============================================================================*/
```

## Writing a custom kernel

### Kernel

A kernel is a `.comp` file, that is, a Vulkan Compute Shader.

### Basic structure

```glsl
/*=============================================================================
 activation.comp:
     O[i] = epilogue(I[i])     (element-wise; epilogue via spec constant)

 - epilogue : 0 none, 1 relu, 2 gelu, 3 silu, 4 sigmoid
 - inPlace : when 1, reads from and writes to the output buffer (in0 unbound)
 - O (typeO) and I (typeI) are fp32 or fp16, accessed through same-binding
   buffer aliases. Dispatch (host): dispatch2(N).
=============================================================================*/
#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require
#include "common/activation.h"
#include "common/spec_constants.h"

layout(local_size_x = 256) in;

layout(constant_id = 0) const uint inPlace = 0;   // 0: separate out, 1: in-place
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;  // O (output) dtype
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;  // I (input) dtype

layout(set = 0, binding = 0) buffer O_BUF_F32 { float bufferO_f32[]; };
layout(set = 0, binding = 0) buffer O_BUF_F16 { float16_t bufferO_f16[]; };
layout(set = 0, binding = 1) buffer I_BUF_F32 { float bufferI_f32[]; };
layout(set = 0, binding = 1) buffer I_BUF_F16 { float16_t bufferI_f16[]; };

layout(push_constant) uniform PushConstants { uint N; };

float readI(uint i) { return (typeI == 0u) ? bufferI_f32[i] : float(bufferI_f16[i]); }
float readO(uint i) { return (typeO == 0u) ? bufferO_f32[i] : float(bufferO_f16[i]); }
void writeO(uint i, float v) { if (typeO == 0u) bufferO_f32[i] = v; else bufferO_f16[i] = float16_t(v); }

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    float x = (inPlace != 0u) ? readO(i) : readI(i);
    writeO(i, applyEpilogue(x));
}
```

The header stamp describes how the host should dispatch the kernel, including the formula, workgroup arrangement, and meaning of the specialization constants. Its purpose is to let readers understand the kernel without opening the host code.

`layout(local_size_x = 256) in;` specifies the number of local invocations that make up one workgroup and their arrangement by dimension. In this case, the workgroup size is (256, 1, 1), and one workgroup consists of 256 invocations in total.

### Shared shader headers

GLSL declarations and functions used by multiple kernels are placed in `vai/shaders/common/`. Shared headers are not independent kernels, so they are not compiled directly and are instead inserted as text into each `.comp` file.

```glsl
layout(constant_id = 0) const uint inPlace = 0;   // 0: separate out, 1: in-place
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;  // O (output) dtype
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;  // I (input) dtype
```

The base path for glslc `#include` directives is `vai/shaders/`. Therefore, a header is referenced in the form `common/<file>.h` regardless of the kernel's actual location.

Only code that is actually shared by multiple kernels is placed in a shared header. Declaring descriptor bindings, push constants, or specialization constants creates a shader ABI that interacts with the host. When a shared header is changed, every kernel that includes that header through `#include` must be verified again.

### Specialization constants - kernel
In Vulkan, kernels are generally compiled through the GLSL -> SPIR-V -> GPU ISA process.

```glsl
#extension GL_GOOGLE_include_directive : require // This extension is required for #include.
#include "common/spec_constants.h"
```

Specialization constants are included in SPIR-V as constants that can be specialized, and their actual values are determined when the pipeline is created. This allows the driver to use those values as constants when generating the final GPU code, making optimizations such as dead code elimination, constant propagation, and loop unrolling easier to perform.

### Descriptors - kernel

Buffers with the same role can be configured to share the same descriptor binding even when their data types differ. In the `activation` kernel, the data type that will actually be used is selected through a specialization constant.

In the example above, if the input data precision is `float`, the `typeI` specialization constant is set to 0 so that values are read from `bufferI_f32`, and if it is `float16_t`, the constant is set to 1 so that values are read from `bufferI_f16`.

### Push constants - kernel

Push constants pass execution parameter values such as shapes, offsets, and small scalar parameters.

The build is automatic. When a kernel is placed in `vai/shaders/`, glslc compiles it. Because `vai/shaders/` is in the `#include` path, `common/*.h` can be referenced through the same path.


## Writing a custom node

Inherit from `vai::Node` and implement `prepare()` and `run()`.

```cpp
class Activation : public vai::Node
{
    eva::ComputePipeline pipe;
    eva::DescriptorSet descSet;

public:
    struct Config {
        vai::Epilogue epilogue = vai::Epilogue::None;
        bool inPlace = false;
        vai::Dtype iDtype = vai::Dtype::None;
        vai::Dtype oDtype = vai::Dtype::None;
    } config_;

    Activation(Config options);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;
};
```

### Constructor

The output slot `out0` and input slot `in0` are registered with `addSlot()` for use in the node graph.

```cpp
inline Activation::Activation(Config options)
: config_(options)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0",  NodeSlot::input);

    config_.iDtype = (config_.iDtype == Dtype::None) ? ioDtypeHint() : config_.iDtype;
    config_.oDtype = (config_.oDtype == Dtype::None) ? ioDtypeHint() : config_.oDtype;
    if (config_.inPlace)   // in-place reads and writes the same buffer
        VAI_ASSERT(config_.iDtype == config_.oDtype);

    pipe = pipelinePool().request("activation", {
        eva::constant_id<0>(config_.inPlace ? 1u : 0u),      // inPlace
        eva::constant_id<SPEC_ID_TYPE_O>(config_.oDtype),    // typeO
        eva::constant_id<SPEC_ID_TYPE_I>(config_.iDtype),    // typeI
        eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)   // epilogue
    });

    descSet = gVaiDestSetPool(pipe.descSetLayout(0));
}
```


#### Specialization constants - node
The specialization constant values are defined in the constructor. They are passed to the kernel as the second argument to `pipelinePool().request()`. The node's `constant_id` values must match the specialization constant `constant_id` values in [Specialization constants - kernel](#specialization-constants---kernel).

### prepare()

```cpp
inline void Activation::prepare()
{
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.dtype() == config_.iDtype);
    VAI_ASSERT(inTen.validShape());

    (*this)["out0"] = config_.inPlace ? inTen : Tensor(inTen.shape(), config_.oDtype);
}
```

The input is validated, and the shapes and data types of the output and intermediate tensors are determined.

### run()

```cpp
inline void Activation::run(eva::CommandBuffer cmdBuff)
{
    auto outBuf = (*this)["out0"].buffer();
    auto inBuf  = config_.inPlace ? eva::Buffer() : (*this)["in0"].buffer();

    descSet.write({ outBuf, inBuf });

    uint32_t N = (*this)["out0"].numElements();

    struct {
        uint32_t N;
    } constants = { N };

    cmdBuff
        .bindPipeline(pipe)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch2(N)
        .barrier(eva::SYNC_SCOPE::COMPUTE_WRITE / outBuf / eva::SYNC_SCOPE::COMPUTE_READ);
}
```

#### Descriptor set - node
The input and output buffers are bound to the kernel descriptors through `descSet.write()`. Each buffer must be connected to match the binding declared in [Descriptors - kernel](#descriptors---kernel).

#### Push constants - node
Push constant values are recorded in the command buffer using `setPushConstants()`. The first argument is the offset, the second is the byte size of the data to record, and the third is the address in memory where the data is stored.

The size, order, and layout of the data passed by the node must match the push constant block declared in [Push constants - kernel](#push-constants---kernel).

#### Dispatch
The dispatch size must match the unit processed by the shader. The `activation` kernel passes the total number of elements to `dispatch2(N)`.

```glsl
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    // Compute output
}
```

Each invocation checks whether `gl_GlobalInvocationID.x` is greater than or equal to the total number of elements, `N`, and terminates immediately if it is outside the valid range.

#### Synchronization

The barrier command after the dispatch makes the current kernel's output visible to the next node's input.

In Vulkan, when multiple kernels are submitted to a single queue, they execute in parallel rather than serially as in CUDA, so explicit synchronization is required to guarantee execution order and memory visibility. Barrier commands are used for this purpose.

### Comparison of passing methods

| Passing method | When passed | Representative values |
|---|---|---|
| Specialization constant | `pipelinePool().request()` | Data type, workgroup size, algorithm branch |
| Push constant | `setPushConstants()` in `run()` | Shape, offset, scalars such as `eps` |
| Descriptor | `descSet.write()` | Input, output, weights, and large parameters |
