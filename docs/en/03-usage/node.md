# Node

![Graph connecting the outputs of three Linear nodes to Attention's Q, K, and V slots](../../assets/node-slots.svg)

A `Node` represents one operation. It receives `Tensor` values through named slots, performs an operation, and passes the results out through slots. A node can record multiple GPU kernels; nodes such as `Identity` and `Flatten` share buffers without recording compute commands.

```cpp
vai::node::Linear linear({.inDim = inDim, .outDim = outDim});
vai::node::Attention attn({
    .headDim = 64,
    .numQHeads = 8,
    .numKVHeads = 2,   // GQA: defaults to numQHeads if omitted
    .kvCacheSize = 1024,
});
```

Each node defines its slots in its constructor. The two node types in the diagram define them as follows:

```cpp
Linear::Linear(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::persistent);
    addSlot("partial", NodeSlot::temporary);   // Allocated only when using the reduce pass
    addSlot("bias", NodeSlot::persistent);
    if (config_.acceptResidual) addSlot("residual", NodeSlot::input);
}
```

```cpp
Attention::Attention(Config config)
: config_(config)
{
    addSlot("out0", NodeSlot::output);
    addSlot("Q", NodeSlot::input);
    addSlot("K", NodeSlot::input);
    addSlot("V", NodeSlot::input);
    addSlot("partialO", NodeSlot::temporary);    // Split-K partial sums, allocated only when kCount > 1
    addSlot("partialML", NodeSlot::temporary);

    // ... Compute Hkv
    if (config_.kvCacheSize > 0)
    {
        addSlot("K_cache", NodeSlot::persistent);
        addSlot("V_cache", NodeSlot::persistent);
        // ... Create cache tensors
    }
}
```

The slot configuration can depend on the options. `Linear` has `residual` only when `acceptResidual` is set. Its `bias` slot always exists and applies only when the weights fill it. Attention's cache slots exist only when `kvCacheSize > 0`. The `partial` slot, omitted from the diagram, is an internal buffer for the operation. It is not used for graph input or output connections.

## 1. Slots

Slots are ports through which `Tensor` values pass. There are four types, each with a different source of values.

| Type | Source of the value (tensor) | Value lifetime |
|---|---|---|
| `input` | Output of a preceding node or external input | Updated on each execution |
| `output` | The node itself (created in `prepare()` and filled by `run()`) | Updated on each execution |
| `persistent` | The user (weights) or the node (caches and constant tables) | Retained across network executions |
| `temporary` | The node itself | Valid only during `run()` |

### 1.1 `input`

Receives the output of a preceding node connected with `-`. At the start of the graph, `NeuralNet` connects it to an external input.

### 1.2 `output`

In `prepare()`, the node chooses the dtype, computes the result shape, and creates a tensor. `run()` records GPU commands to fill that buffer, which holds the result when execution finishes.

### 1.3 `persistent`

These slots retain values between executions. There are two cases.

**1. The user prepares weights**: These slots hold learned parameter values, such as `Linear`'s `weight` and `bias`. If they are not prepared, shape validation in `Node::prepare()` fails.

```cpp
vai::Checkpoint ckpt(VAI_DATA_DIR "/ckpts/mnist/jroot");
const vai::Dtype wDt = net.config().wDefaultDtype;   // the net's weight dtype
linear["weight"] = vai::Tensor(ckpt["fc.weight"], wDt);
linear["bias"]   = vai::Tensor(ckpt["fc.bias"], wDt);
```

**2. The node prepares other values itself**: Examples include Attention's `K_cache` and `V_cache`, and Rope's `freqs`.

```cpp
// Attention constructor: the node creates the cache itself.
(*this)["K_cache"] = Tensor({Hkv, config_.kvCacheSize, config_.headDim}, config_.kvCacheDtype);
```

Whether the value is fixed, as with weights, or accumulates at each step, as with a KV cache, `persistent` buffers remain alive between executions. `input` and `output` buffers are updated on each execution and reused through the buffer pool.

### 1.4 `temporary`

These are workspace buffers used only inside an operation. They may be needed only on a particular execution path, as with `Linear`'s `partial`; no buffer is allocated when that path is not used.

## 2. Working with slots

### 2.1 Slot names

Slots are declared in the constructor with `Node::addSlot`. Users cannot add new names from outside the node, and accessing a nonexistent name throws an exception.

```cpp
linear.slot("weight");   // OK
linear.slot("scale");    // Slot not present in Linear (exception)
```

The per-node slot tables in [Built-in nodes](../06-nodes.md) list the available slots.

### 2.2 Reading values and accessing slots

`Node::operator[]` and `Node::slot()` have similar names but return different objects.

```cpp
linear["weight"];       // Tensor&    Read and store the value
linear.slot("out0");    // NodeSlot&  Used for connections
```

The `-` operator calls `slot()` internally. Use `[]` for values and `slot()` for graph connections.

## 3. Connections

The `-` operator connects an `output` slot of the node on the left to an `input` slot of the node on the right. Use `/` with a name to choose the slot.

```cpp
conv/"out0" - "in0"/flatten/"out0" - "in0"/linear;
```

### 3.1 Selecting slots

The `/` slot selection operator chooses an `input` slot on the left side of a node and an `output` slot on its right. This follows the left-to-right direction of data flow.

```
          "in1" / add1 / "out0"
          └┬──┘          └─┬──┘
          input slot    output slot
```

When `/` is omitted, the slot before the node defaults to `in0`, and the slot after the node defaults to `out0`.

| Notation | `input` slot | `output` slot |
|---|---|---|
| `node` | `in0` | `out0` |
| `"X"/node` | **`X`** | `out0` |
| `node/"Y"` | `in0` | **`Y`** |
| `"X"/node/"Y"` | **`X`** | **`Y`** |

The connection above can therefore be shortened to:

```cpp
conv - flatten - linear;
```

Only `input` slots can appear with `/` before a node, and only `output` slots can appear after it. `persistent` and `temporary` slots are not used for graph connections.

### 3.2 Nodes with multiple slots

Nodes with multiple inputs need explicit slot selection. For example, `Add` receives `in0` and `in1`; using only `-` would send both connections to `in0`, so one slot must be specified.

```cpp
input - norm1 - mhAttn - "in0"/add1;   // Attention result goes to add1's in0
input                  - "in1"/add1;   // Original input goes to add1's in1
```

This is a residual connection. The output of `input` branches to two destinations; connecting one output to multiple inputs is allowed.

Normally, an input slot connects to only one output slot. As an exception, the output slots of Gate's `then` and `else` paths can connect to the same input slot. Only the selected path is recorded in the commands, so only that path's result is used during execution.

```cpp
gate/"then" - relu        - "in0"/merge;
gate/"else" - passthrough - "in0"/merge;
```

Some nodes use entirely different slot names. `Attention` receives `Q`, `K`, and `V`, so all three must be specified.

```cpp
input - projQ - splitQ - perQ - "Q"/attn - mergeOut - projOut;
input - projK - splitK - perK - "K"/attn;
input - projV - splitV - perV - "V"/attn;
```

## 4. Node lifecycle

A node has a two-stage lifecycle. `NeuralNet` calls these methods; they are not called directly.

|                      | When | Role |
| -------------------- | ------------- | --------------------- |
| `Node::prepare()`    | When the input shape is known | Validate shapes and dtypes, and prepare output tensors, workspace tensors, and pipelines |
| `Node::run(cmdBuff)` | When recording the graph | Record GPU commands in the command buffer |

`Node::run()` records commands rather than executing the operation. The actual computation happens later when the GPU executes those commands. This is why vAi does not traverse the graph again for every inference. Once recorded, the command buffer is reused while the recording conditions remain the same.

When the input shape changes, `Node::prepare()` is called again. Nodes that select kernels based on shape choose their pipelines again at this point. For details, see [prepare() in Writing custom nodes and kernels](../contributing/nodes.md#prepare).
