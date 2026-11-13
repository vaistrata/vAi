# Tensor

A Tensor is data that flows between nodes.

Most tensors are held in the [Slots of nodes](node.md#slot) and they move from the output slot of the previous node to the input slot of the next node, flowing through the graph. This tensor is usually created by the node itself, but users can also create it directly and put it into the model.

A node's slot is `Node::slot()` and the tensor contained in that slot is `NodeSlot::getValueRef()`. `Node::operator[]` shortens these two calls into one, and accesses the tensor directly by the slot name.
(see [Reading values and getting slots in Node](node.md#reading-values-and-getting-slots)).

```cpp
linear.slot("out0")                 // NodeSlot& — The slot itself. Used for graph connections
linear.slot("out0").getValueRef()   // Tensor&   — The tensor contained in that slot
linear["out0"]                      // Tensor&   — Same as the line above
```

## Data and handle

The inside consists of two classes.

```cpp
// The owner of the actual bytes. Shared by tensors.
class TensorData
{
    std::vector<char> hostData_;    // Host Data
    const Dtype dtype_;
    vai::PooledBuffer buffer_;      // Device Buffer
};

class Tensor
{
    std::shared_ptr<TensorData> data_;    // Copying only increases this reference
    std::vector<uint32_t> shape_;
};
```

The actual bytes and dtype are owned by `TensorData`. Bytes can be in host memory, a GPU buffer, or both. The `Tensor` handled by the user is a handle that points to `TensorData`, and the shape is held separately by each handle.
Therefore, copying creates another handle that points to the same data, and the two handles can see the same data in different shapes.

```cpp
Tensor a({M, N});               // fp32, M x N
Tensor b = a;                   // Not a duplication, but another handle that points to the same TensorData
Tensor c = a.reshape({M * N});  // Only c is 1-dimensional. For a, [M, N] remain the same, and the data is still one
```


## State

A Tensor may or may not have a shape, host data, and a device buffer, respectively. That combination is the state of the Tensor.

| State | Meaning | Confirmation |
|---|---|---|
| Empty Handle | No shape or data. Initial State of the Slot | `validShape() == false` |
| Only Shapes | The shape and dtype have been determined, but there are no bytes yet. | `validShape() == true` |
| Host Data | The bytes are in host memory | `hasHostData() == true` |
| Device Buffer | The bytes are in a GPU buffer. Target read and written by the node's `run()` | `gpuBufferBound() == true` |

The latter two are not exclusive. Right after the upload, it can be on both sides, and `NeuralNet` uploads the weights and clears the host side.

The states a Tensor goes through and the order in which it goes through them are different for each Tensor. The four paths are addressed in [Representative Tensor Flows And Lifecycles(#Representative-Tensor-Flows-And-Lifecycles).

## Creation

There are three ways to create a tensor.

### 1. By shape

```cpp
Tensor x({M, N});                   // fp32 is basic
Tensor y({H, W}, Dtype::Fp16);
```

It starts with only the shape and dtype determined, with no data yet. **dtype cannot be changed after creation**. As it is an attribute attached to the data, if you need a different dtype, you need to create a new tensor.

The purpose is to fill in the data later and create the tensor in advance. Input data, table data, the output slot of a node, etc. are created in this way. The data is later filled in directly by the function in [Host data upload](#host-data-upload) or filled in by the GPU like a node output.

### 2. From safetensors

This method is used to upload weights. At this point, you need to pay attention to dtype.

```cpp
SafetensorsParser parser(VAI_DATA_DIR "/ckpts/mnist/jroot/model.safetensors");
Tensor w(parser["fc.weight"]);                  // The dtype as it is in the file
Tensor w(parser["fc.weight"], Dtype::Fp16);     // Convert to fp16 and load
```

The shape is taken as recorded in the file, and it starts with the host data filled in. Given the second argument, it converts the dtype while loading. BF16 is not supported yet, so if it is not specified, it is resolved to fp32?? (Lossless)

In vAi, you need to explicitly specify the dtype of the tensor for each slot. It won't cast automatically, and if casting is needed, you need to explicitly insert a casting node.
It is the user's responsibility to insert the right tensor if the node requires a specific dtype. If a different type is inserted, a runtime error may occur or the behavior may become undefined during execution.

Weights take their dtype from the tensor itself: a node reads the dtype of the weight it holds. Load weights at the net's `wDefaultDtype` and norm coefficients at fp32.

```cpp
linear["weight"] = Tensor(parser["fc.weight"], net.config().wDefaultDtype);  // the net's weight dtype
linear["weight"] = Tensor(parser["fc.weight"]);   // the file's dtype, whatever it is
```

The shape is not specified like dtype, but the shape itself recorded in the file becomes the tensor's shape. That shape could be different from the shape that the node wants. In that case, you need to fix it on the host before putting it into the slot.
(see [Host-only tensor calculation](#host-only-tensor-calculation)).

The tensor created in this way is still in the [Host data](#state) state. Putting it into the slot does not mean that it goes directly to the GPU. Instead, at the preparation stage of the first forward, `NeuralNet` uploads it to the device buffer. (see [Weights in Representative tensor flows and lifecycles](#1-weights)).

### 3. From an existing tensor

Copying and assignment, `Tensor::reshape`, create a new tensor and share the data with the original.
However, calculations that change the memory layout, like `Tensor::permute`, create new data.
(see [Host-only tensor calculation](#host-only-tensor-calculation)).

```cpp
Tensor b = a;                    // New tensor — shares data with a
Tensor c = a.reshape({M * N});   // New tensor — shares data with a, but has a different shape
Tensor t = a.permute(1, 0);      // New tensor — owns newly rearranged data
```

## Host data upload

There are two functions that upload data to a tensor with only a shape.`Tensor::setFloats()` receives a `std::vector<float>` and converts each element to the tensor's dtype before uploading it.
`Tensor::setData<T>()` receives a raw pointer of an arbitrary type and copies the bytes as they are without conversion. However, the user needs to guarantee the following for `setData()`.

- `sizeof(T)` needs to be the same as the size of an element of the tensor.
- The pointer needs to point to valid data for as many elements as the tensor has.

```cpp
// Image Input — convert float values to the tensor's dtype and store them.
std::vector<float> pixels(28 * 28);       // Pixels normalized to [0, 1]
Tensor img({1, 28, 28}, Dtype::Fp16);
img.setFloats(pixels);                    // Convert fp32 -> fp16

// Token Input — memcpy without conversion. The bytes of the integer IDs are put in as they are.
std::vector<int> tokenIds = tokenizer.encode(prompt);
Tensor tokens({(uint32_t)tokenIds.size()});
tokens.setData(tokenIds.data());          // sizeof(int) == size of an fp32 element
```

The uploaded data is still only in the host ([Host data](#state) State). The user's responsibility ends here, and `NeuralNet` automatically moves it to the GPU during the forward pass.

## Host-only tensor calculation

Sometimes, the shape or layout of the tensor included in the checkpoint could be different from what the node wants. In such cases, fix them using the following calculations while in the Host Data state, before being uploaded to the GPU.

All four calculations leave the original as it is and return a new tensor. However, it varies by calculation whether the result tensor owns its own new data or shares data with the original. The last column of the table shows the distinction.

| Calculation | Tasks | Data |
|---|---|---|
| `Tensor::permute` | Axis Rearrangement (e.g., Transpose) | Owns new data |
| `Tensor::split` | Cut the leading axis into multiple tensors | Each piece owns new data |
| `Tensor::concat` | Concatenate along the leading axis | Owns new data |
| `Tensor::reshape` |Same data in different shapes (same number of elements) | Shares with the original |

```cpp
// As c_attn in HF GPT-2 is saved as [in, out], it needs to be transposed
projQKV["weight"] = Tensor(parser["attn.c_attn.weight"], dt).permute(1, 0);

// Divide the fused QKV into three
auto [q, k, v] = Tensor(parser["attn.c_attn.weight"], dt).split(D, 2 * D);

// To turn the two projections, SwiGLU: gate and up, into one GEMM, concatenate them.
gateValue["weight"] = Tensor(parser["gate.weight"], dt).concat(Tensor(parser["up.weight"], dt));
```

`permute`, `split`, and `concat` operate only when host data is available. It is the node's task to fix the tensor after the device buffer has been attached. (e.g.: `Reshape`,
`Permute`).

## Reading output

Forward is asynchronous. If you call `Tensor::downloadToHost()` on the output tensor filled by `model(...)`, it waits for the GPU to complete and downloads the result to the host. Downloading occurs only once, on the first call.

`Tensor::getData<T>()` calls `downloadToHost()`, reinterprets the pointer of the host data as `const T*`, and returns it. As it does not convert the values, `T` should be the type of the values that are actually stored.

```cpp
Tensor out;
net(inDict, {{"nextTokenId", &out}});
uint32_t next = out.getData<uint32_t>()[0];   // From inside downloadToHost()
```

As the output buffer is overwritten in place by the next forward, the result needs to be read before the next `model()` call. If you call `downloadToHost()` using a handle that has passed that point, the data has already been corrupted, and it will trigger an assert.

You don't necessarily need to move the output to the host. You can pass the output tensor directly as the input to another `NeuralNet`, or pass it as the next input of the same `NeuralNet`, as in autoregressive generation.
Input that already has a device buffer is read directly by the graph, resulting in a **zero-copy** connection that does not pass through the host. This method is used to concatenate multiple models.

## Representative tensor flows and lifecycles

Even if they pass through the same states, the path they take differs depending on which tensor they are.

### 1. Weights

It begins at the host, is uploaded to the device, and then remains fixed.

```
safetensors Load (Host Data)
  → Host-Only Tensor Calculation (Transpose, etc., if necessary)
  → Put in the persistent slot
  → At the preparation stage of the first forward, NeuralNet uploads it (Device Buffer)
```

As the slot is `persistent`, the buffer won't be released after the forward is completed and remains on the GPU. Therefore, as long as `NeuralNet` is alive, you don't need to upload the weights again.

### 2. External input

It starts at the host in the same way, but the values are newly filled in for each execution.

```
Creation via Shape
  → setData / setFloats (Host Data)
  → model() call (uploaded in the slot of the input node)
  → Fill the values again before the next call
```

Typically, instead of creating a new tensor, you change only the values with `setData` on the same handle and pass it again when the same shape is repeated, like in a decode loop. The efficiency might decrease if you create a new tensor and pass it through every time, because **command buffer re-recording would occur**.

### 3. Node output

It is not a tensor that the user creates or passes through. It does not pass through the host, but is created directly inside the GPU, moves to the next node, and disappears after being used.

```
prepare() only creates the shape
  → NeuralNet attaches a buffer from the buffer pool (Device Buffer).
  → GPU fills
  → To the next node's input slot (handle only is passed)
  → The used buffer goes back to the pool and is reused.
```

### 4. forward output

It is the tensor that `model(...)` fills through the output argument. You can download it to the host and read it, or you can pass it directly as the input to a different forward.

```
model() returns
  → Reading: use downloadToHost() / getData<T>() to wait for the GPU to complete and download the data (Host Data)
  → Passing: directly to the input of a different forward (zero-copy without passing through the host)
  (Either way, before the next forward overwrites the same buffer)
```
