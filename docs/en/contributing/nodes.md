# Writing custom nodes and kernels

A `Node` is generally a unit of computation in the graph. It prepares the required `Tensor` objects and records GPU commands.

GPU commands specify the kernel to run, the buffers it reads and writes, and the range of data to process. These commands are recorded in a command buffer (`CommandBuffer`). The vAi engine submits this command buffer to the GPU's execution queue. The GPU runs the kernel according to the submitted commands.

Adding a GPU operation requires C++ `Node` code and GLSL kernel code. The `Node` code prepares `Tensor` objects and records the settings and commands needed to run the kernel. The kernel code implements how the GPU reads values from buffers, computes results, and writes them back. The tensor shapes and dtypes, buffer binding numbers, and data range specified by the `Node` must match the kernel's declarations and computation.

Before implementing an operation, check the [built-in nodes](../06-nodes.md). An operation that can be expressed by combining existing nodes can be assembled as a [group](../03-usage/node-group.md). If new GPU computation is needed, follow the steps in this document to write a `Node` and kernel. A single `Node` can also run multiple kernels. This chapter follows the implementation of the [`Activation` node](../../../vai/nodes/activation.h) and the [activation.comp kernel](../../../vai/shaders/activation.comp) to explain how to write custom nodes and kernels. The examples select ReLU as the activation function and use the path that writes results to a separate output.


## 1. Define the operation's inputs and outputs

First, define the values to compute and the supported input conditions. These conditions apply to input validation in the `Node`, kernel implementation, and result verification.

| Item | What to define | `Activation` example |
|---|---|---|
| Computation | How to compute the output from the input | ReLU: apply `max(x, 0)` to each element |
| Slot names | Inputs and outputs to connect in the graph | Input `in0`, output `out0` |
| Shape | Meaning of each axis and supported sizes | Input and output have the same shape |
| dtype | Supported input and output storage formats | FP32 and FP16 |
| Configuration | Values to set when creating the `Node` | Activation function `epilogue`, in-place operation `inPlace` |

Document any restrictions on rank, dimension sizes, or alignment. For example, an operation that normalizes only the last dimension should specify that axis and the size of its weights.

Configuration is set when the `Node` is created. The actual input shape and `dtype` are read in `prepare()`. The output `dtype` is not part of the node's configuration. `prepare()` takes the net's default output `dtype` from `oDefaultDtype()`. When one node has to write another `dtype`, the model (`Net`) sets it in its constructor with `setODtype()`, and the node reads it as `oDtype_`. A weight's `dtype` is not configured either: the node reads it from the `Tensor` in the slot.

The following example declares a node configured for ReLU as a model member. It leaves `inPlace` at its default value of `false`.
```cpp
vai::node::Activation activation{{.epilogue = vai::Epilogue::Relu}};
```

## 2. Prepare tensors in the node

### 2.1 Register slots and store configuration in the constructor

A custom node class publicly inherits from `vai::Node` and implements `prepare()` and `run()`. The node header includes `vai/nodes/node-common.h`. For how to create a node as a model member and connect it to the graph, see [Writing custom models](models.md).

The following code is taken from the declaration and constructor of `Activation`. The constructor stores the configuration and registers slots for inputs and outputs. This code is in the `vai::node` namespace.

```cpp
class Activation : public vai::Node
{
    // ... pipeline and internal state
public:
    struct Config {
        vai::Epilogue epilogue = vai::Epilogue::None;
        bool inPlace = false;
    } config_;

    Activation(Config config);
    void prepare() override;
    void run(eva::CommandBuffer cmdBuff) override;

    // ...
};

inline Activation::Activation(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
}
```

Register each slot according to its purpose.

| Kind | Purpose |
|---|---|
| `input` | A `Tensor` received from a preceding node |
| `output` | A `Tensor` passed to a following node |
| `temporary` | A temporary `Tensor` used while computing this node |
| `persistent` | A `Tensor` retained between executions, such as weights or a cache |

The shape and dtype of a `persistent` tensor are set in the constructor or weight loader. Before the first `prepare()`, the engine attaches GPU buffers and uploads values from the CPU. A cache that requires initial values must be filled before use.

<a id="prepare"></a>

### 2.2 Validate inputs and configure outputs in `prepare()`

`prepare()` reads the inputs from their slots and checks the conditions defined in section 1. It uses those inputs to determine the shapes and dtypes of output and temporary tensors. It also checks the input conditions needed to select a pipeline. Pipelines are covered in section 4.

The following is the tensor preparation part of `Activation::prepare()`. Use this pattern to implement `prepare()`.

```cpp
inline void Activation::prepare()
{
    const Dtype oDtype = (oDtype_ == Dtype::None) ? oDefaultDtype() : oDtype_;
    const Tensor& inTen = (*this)["in0"];
    VAI_ASSERT(inTen.validShape());

    const Dtype iDtype = inTen.dtype();
    if (config_.inPlace)   // in-place reads and writes the same buffer
        VAI_ASSERT(oDtype == iDtype);

    // ... prepare a pipeline for the input dtype

    (*this)["out0"] = config_.inPlace ? inTen : Tensor(inTen.shape(), oDtype);
}
```

Here, `(*this)["slot name"]` reads or assigns the `Tensor` in a slot. After tensor properties are specified, as in `Tensor(shape, dtype)`, the engine attaches the GPU buffers needed for output and temporary tensors, then calls `run()`. The new output buffer is used in `run()`.

The default path creates a new output tensor with the same shape as the input.

## 3. Compute values in the kernel

A kernel is a GLSL compute shader written in a `.comp` file. First, decide what one GPU thread will process. Choose a unit suited to the operation, such as one element, one row, or one tile. The dispatch range supplied by the `Node` must match that unit. The following excerpt from `activation.comp` shows the workgroup size and computation. This kernel assigns one element to each thread.

```glsl
#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_16bit_storage : require
#include "common/activation.h"
#include "common/spec_constants.h"

layout(local_size_x = 256) in;

// ... buffer and dtype configuration declarations
layout(push_constant) uniform PushConstants { uint N; };

// ... readI(), readO(), writeO() for dtype-specific reads and writes
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= N) return;

    float x = (inPlace != 0u) ? readO(i) : readI(i);
    writeO(i, applyEpilogue(x));
}
```

Each thread in this kernel computes one element. `local_size_x = 256` places 256 threads in a workgroup. Each thread uses `gl_GlobalInvocationID.x` as the index `i` of the element to process. `N` is the total number of elements. Some threads in the last group may have no element to process, so the following condition exits threads outside the valid range.

```glsl
uint i = gl_GlobalInvocationID.x;
if (i >= N) return;
```

On the default path, the kernel reads the input with `readI()`, computes the result with `applyEpilogue()`, and writes it to the output with `writeO()`. With ReLU selected, `applyEpilogue(x)` computes `max(x, 0.0)`. Buffer read and write functions should use the storage format for the dtype.

An operation in which multiple threads write to the same output location must handle write conflicts. In `Activation`, each thread computes a different output element, so no conflict handling is needed. In the `Argmax` kernel, however, multiple threads compute a single output value together. If they write to the same output location, they can overwrite one another's results. To prevent this, combine the intermediate results from each thread and have one thread write the final value. When intermediate results are shared within a workgroup, synchronize inside the kernel so that values are read only after they are all ready. `Argmax` combines its intermediate results and then lets only thread 0 in the workgroup write the row's result to `outIdx[row]`.

Each thread in `Argmax` stores its maximum value and index in shared memory. After `barrier()` ensures that all required values are ready, they are compared. Once the comparisons are complete, only thread 0 writes the final result to the output.

```glsl
sMax[tid] = bestVal;
sIdx[tid] = bestIdx;
barrier();

// ... compare intermediate results. Call barrier() at each comparison step as well.

if (tid == 0u) outIdx[row] = sIdx[0];
```
GLSL's `barrier()` synchronizes threads within the same workgroup. The `.barrier(...)` described in section 5 synchronizes buffer access between GPU commands. Here, `barrier()` coordinates when intermediate results can be read, and the final `if` statement selects a single thread to write the result.

## 4. Match the node and kernel interfaces

To compute results, the kernel needs to know which buffers to use, their data formats, and the size of the computation. These are passed as follows.

| Information | How it is passed | What must match between the node and kernel |
|---|---|---|
| Input, output, and weight buffers | Descriptors: assign buffers to kernel binding numbers | `set`, `binding`, buffer role |
| Values set when creating a pipeline | Specialization constants | Constant ID, value type, and meaning |
| Small values recorded in a command | Push constants | Field order, size, and byte offset |

Push constants hold shapes, offsets, small scalar values, and similar data. Specialization constants hold values that configure the kernel, such as dtypes or algorithm selections. Choose the mechanism according to when the value is set.

### 4.1 Request a pipeline for the input conditions

A pipeline is a GPU execution object that combines a kernel with its specialization settings. Pass the kernel name and specialization constants to `pipelinePool().request()`. The pool reuses pipelines for identical requests. The kernel name passed to `pipelinePool().request()` omits the file extension. For example, request `activation.comp` as `"activation"`.

The following part of `Activation::prepare()` prepares the pipeline. The input dtype is read from the actual input. The output dtype is the `oDtype` computed above.

```cpp
const PipeParams p = { .iDtype = iDtype };
if (!pipe || p != pipeParams)
{
    pipeParams = p;
    pipe = pipelinePool().request("activation", {
        eva::constant_id<0>(config_.inPlace ? 1u : 0u),
        eva::constant_id<SPEC_ID_TYPE_O>(oDtype),
        eva::constant_id<SPEC_ID_TYPE_I>(iDtype),
        eva::constant_id<SPEC_ID_EPILOGUE>((uint32_t) config_.epilogue)
    });
    if (!descSet)
        descSet = gVaiDestSetPool(pipe.descSetLayout(0));
}
```

`PipeParams` is a struct defined inside this node. It stores the input dtype used to select the pipeline. When that condition changes, the pipeline is requested again. A new node must also compare every shape or dtype condition used in pipeline selection.

`descSet` is a descriptor set containing the list of buffers to bind to the kernel. Obtain it from the shared `gVaiDestSetPool`, as existing nodes do. This example reuses the set because it uses the same kernel buffer layout. When switching to a different kernel, obtain a new set for that layout if the buffer types, counts, or binding numbers differ.

### 4.2 Match C++ and GLSL buffers and constants

The following declarations in `activation.comp` receive the in-place setting, dtypes, and buffers.

```glsl
layout(constant_id = 0) const uint inPlace = 0;
layout(constant_id = SPEC_ID_TYPE_O) const uint typeO = 0;
layout(constant_id = SPEC_ID_TYPE_I) const uint typeI = 0;

layout(set = 0, binding = 0) buffer O_BUF_F32 { float bufferO_f32[]; };
layout(set = 0, binding = 0) buffer O_BUF_F16 { float16_t bufferO_f16[]; };
layout(set = 0, binding = 1) buffer I_BUF_F32 { float bufferI_f32[]; };
layout(set = 0, binding = 1) buffer I_BUF_F16 { float16_t bufferI_f16[]; };
```

Values passed by the node map to the following locations. Check both the numbers and types.

| Value specified in C++ | Location in the kernel |
|---|---|
| First buffer in `descSet.write({outBuf, inBuf})` | `set = 0, binding = 0`: output |
| Second buffer in the same list | `set = 0, binding = 1`: input |
| `constant_id<0>(...)` | `constant_id = 0`: whether to operate in place |
| `constant_id<SPEC_ID_TYPE_O>(...)` | `constant_id = SPEC_ID_TYPE_O`: output dtype |
| `constant_id<SPEC_ID_TYPE_I>(...)` | `constant_id = SPEC_ID_TYPE_I`: input dtype |
| `constant_id<SPEC_ID_EPILOGUE>(...)` | `epilogue` in `common/activation.h`: activation function |
| Push constant `uint32_t N` | `uint N` in `PushConstants` |

The dtype constants use the shared IDs and values in [common/spec_constants.h](../../../vai/shaders/common/spec_constants.h). `activation.comp` declares FP32 and FP16 buffers at the same binding numbers, then uses the dtype constants to select which to use. This preserves the order in which C++ binds buffers. The constant that selects the activation function is declared in `common/activation.h`. Do not redeclare the same ID in the kernel.

When passing multiple fields as push constants, the byte layout of the C++ struct and GLSL block must also match. Arrays and vectors in particular can have different offsets due to alignment, even if they are declared in the same order.

## 5. Record execution commands

`run()` records GPU commands using the prepared buffers and pipeline. GPU computation takes place after the engine submits these commands.

The following code is taken from `Activation::run()`.

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

Here, `.buffer()` returns the GPU buffer range used by a `Tensor`. Pass this range directly to descriptors and barriers. This addresses the correct location even when the tensor refers to part of a larger buffer.

The arguments to `setPushConstants()` are the byte offset, data size, and data address, in that order. The example records the element count `N` at offset 0.

For in-place operation, set `inPlace` in `Config` to `true` when creating the node in the model. This path uses the same `Tensor` for input and output and writes each result to the location read by `readO()`. The second buffer is left empty because the input is also read from the output buffer. Input and output dtypes must match. Also check whether other nodes use the original input values.

### 5.1 Work units and dispatch ranges

Pass the number of threads in each dimension to `dispatch2(x, y, z)`. This function computes the group count by dividing by the pipeline's workgroup size and rounding up. Pass workgroup counts directly to `dispatch(x, y, z)`.

The example uses `dispatch2(N)` because it needs one thread per element. For a kernel that computes by row or tile, set the dispatch range to match that unit.

### 5.2 Synchronize buffers between kernels

The `Activation` kernel writes its results to `outBuf`. Before the next kernel reads them, record a barrier (`.barrier`) so that the writes are complete and their results are available for reading.

```cpp
.barrier(
    eva::SYNC_SCOPE::COMPUTE_WRITE /  // The preceding kernel writes values.
    outBuf /                         // Buffer range to synchronize.
    eva::SYNC_SCOPE::COMPUTE_READ);   // The next kernel reads values.
```
When a single node runs multiple kernels, record the required barriers between a kernel's writes and the next kernel's reads of those results. If the next operation is a copy rather than computation, select a synchronization scope for copy reads instead of `COMPUTE_READ`.

### 5.3 Reuse commands and update state on each call

The engine reuses recorded GPU commands. It calls `prepare()` and `run()` when recording or rerecording commands. Push constant values therefore remain unchanged while the recorded commands are reused.

Counters or state that change on each call can be passed through a separate buffer. For CPU updates, wait for the previous GPU work to finish, then update the values in the model's execution method. This flow is covered in [State management in the execution method](models.md#42-managing-state-in-execution-methods).

An API that changes configuration after construction must also update the recorded commands. If command values or buffer bindings change, call `invalidateCommandBuffer()` on the owning model to request rerecording. If specialization settings change, select the pipeline again as well.

## 6. Build and verify results

### 6.1 File placement and kernel builds

| Purpose | Node | Kernel |
|---|---|---|
| Operations used by multiple models | `vai/nodes/<name>.h` | `vai/shaders/<name>.comp` |
| Operations used by a specific model | `models/<impl>/<model>/nodes/<name>.h` | `models/<impl>/<model>/shaders/<name>.comp` |

The [root build configuration](../../../CMakeLists.txt) collects shared kernels and builds them with `vAi`.
Request them from the node with `pipelinePool().request("kernel name", {...})`, as in the example.
Include the node header in the C++ file that uses it.

The [model build configuration](../../../models/CMakeLists.txt) builds model-specific kernels under the `<impl>-<model>-shaders` target. `<impl>` is `production` or `experimental`.
The directory the compiled kernels land in is passed through the `<IMPL>_<MODEL>_SHADER_OUTPUT_DIR` macro, and the node reads its kernel from there.
The following code is taken from [YOLOX's SPPF node](../../../models/experimental/yolox/nodes/sppf.h).

```cpp
inline eva::SpvBlob sppfSpv = eva::SpvBlob::readFrom(EXPERIMENTAL_YOLOX_SHADER_OUTPUT_DIR "/sppf.comp.spv");
```

Also add a build dependency to the executable target that uses the kernel.
The following code is from the [YOLOX demo build configuration](../../../models/experimental/yolox/CMakeLists.txt).
Here, `${PROJECT_NAME}` is `demo-yolox`.

```cmake
add_dependencies(${PROJECT_NAME} experimental-yolox-shaders)
```

Place GLSL code shared by multiple kernels in `vai/shaders/common/`.
Include headers from this directory with `#include "common/<name>.h"`.
If buffer declarations or constant layouts in a shared header change, check the kernels that use it as well.

Check the following before submitting.

- Run the supported combinations of shapes and dtypes.
- Check the minimum size and inputs that are not multiples of the workgroup size. For the example kernel, element counts of 1, 256, and 257 cover these boundaries.
- Check boundary values for the selected computation. For ReLU, compare results for negative values, zero, and positive values.
- Run the same graph repeatedly and compare results after changing the input shape as well.
- If there are paths for shared outputs, in-place operation, or temporary tensors, check those paths too.
- Record the inputs, reference implementation, dtype, tolerance, and execution method used for comparison.

A PR that adds a node must include Python bindings and tests. The PR checks fail if either is missing. See [7. Files to include in the PR](#7-files-to-include-in-the-pr) for instructions.

### 6.2 Code and comment conventions

| Item | Convention |
|---|---|
| Node declaration | Publicly inherit from `vai::Node`. A `class` declaration is allowed. |
| Namespace | Place built-in nodes in `vai::node`. Place model-specific nodes in the model's namespace. |
| C++ names and formatting | Use `PascalCase` for types and `camelCase` for functions and variables. Put a function's opening brace on the next line. Follow the [common code conventions](../index.md#3-common-code-conventions). |
| Header definitions | Use header guards. Add `inline` to regular member functions defined outside the class. |
| Node comments | Describe the name, computation, input/output/weight slots and shapes, and supported conditions. |
| Kernel comments | Describe the formula, buffer roles, meanings of specialization constants, and work unit. Choose formatting for the rest of the kernel to suit the implementation. |

Use the following format for a node's documentation comment.
The first line contains the actual class name and a summary. If there are weights, add a `Params:` entry as well.
The following example is taken from the input and output section of the `Activation` comment.

```cpp
/*=============================================================================
  Activation - Applies an element-wise activation.
  Input: in0 [any shape]
  Output: epilogue(in0); optionally in-place

  // ... supported conditions and behavior
=============================================================================*/
```

Use a block comment with the same delimiters for a kernel.
The following example is taken from the comment in `activation.comp`.

```glsl
/*=============================================================================
 activation.comp:
     O[i] = epilogue(I[i])     (element-wise; epilogue via spec constant)

 // ... option descriptions
 - O (typeO) and I (typeI) are fp32 or fp16, accessed through same-binding
   buffer aliases. Dispatch (host): dispatch2(N).
=============================================================================*/
```

## 7. Files to include in the PR

A PR that adds a `Node` must include the following files in addition to the `Node` and its kernel.

| Order | File | Location |
|---|---|---|
| 1 | Python binding | `python/bindings/nodes/<name>.cpp` or `python/bindings/models/<model-name>/nodes/<name>.cpp` (without the `<impl>` folder) |
| 2 | Name export (common `Node` only) | `python/pyvai/nodes.py` |
| 3 | Test | `python/tests/nodes/test_<name>.py` or `python/tests/models/<model-name>/test_<name>.py` |
| 4 | Node documentation | `docs/ko/06-nodes.md`, `docs/en/06-nodes.md` |

[`test_node_completeness.py`](../../../python/tests/test_node_completeness.py) fails if any binding, registration, or test is missing.

### 7.1 Python binding

List the `Config` fields in the binding file and register the binding on the last line. Do not modify lists in other files.
The following excerpt is from the [`Slice` binding](../../../python/bindings/nodes/slice.cpp).

```cpp
static void bindSlice(nb::module_& m)
{
    using C = vai::node::Slice::Config;
    bindNode<vai::node::Slice>(m,
                               "SliceNode",
                               "Take `count` consecutive indices ...",
                               required("start", &C::start),
                               field("count", &C::count).check(atLeast(1u), "must be >= 1"),
                               field("axis", &C::axis));
}
PYVAI_NODE_BINDING(bindSlice);
```

| Function | Meaning |
|---|---|
| `field(name, &C::member)` | The default is the value from `Config{}`. Do not repeat defaults in the binding. dtype fields default to `None` in Python; `None` leaves the `Config{}` value unchanged. |
| `field(name, &C::member, value)` | The binding specifies the default value. |
| `required(name, &C::member)` | No default value. |
| `.check(condition, message)` | Raises `ValueError("<Node>: <name> <message>")` if the condition is false. Prevents invalid arguments from reaching the engine's `VAI_ASSERT`. |
| `PYVAI_NODE_BINDING(function)` | Registers a common `Node` binding. |
| `PYVAI_MODEL_NODE_BINDING(model-name, function)` | Registers a model-specific `Node` binding. |

See the comments in [`bind-node.h`](../../../python/bindings/nodes/bind-node.h) for details.

### 7.2 Name export

For a common `Node`, add the name to the import list and `__all__` in [`python/pyvai/nodes.py`](../../../python/pyvai/nodes.py).

### 7.3 Tests

Tests call `check_node()` with a function that creates the `Node`, input shapes, a reference implementation written with `torch`, and a tolerance.
The following excerpt is from the [`Slice` test](../../../python/tests/nodes/test_slice.py).

```python
def test_slice_axis0_block():
    check_node(
        lambda: nodes.SliceNode(start=1, count=2, axis=0),
        {"in0": (4, 5)},
        lambda in0: in0[1:3],
        tol=EXACT,
    )
```

`check_node()` checks that the output matches the reference implementation within tolerance, remains the same on repeated execution, and contains no `NaN` or `Inf`.
See [`node_check.py`](../../../python/tests/node_check.py) for tolerance types and selection criteria.

### 7.4 Node documentation

Regenerate the documentation as described in [4.4 of Contribution workflow and automated checks](README.md#44-node-documentation), and commit both generated files with the `Node`.
