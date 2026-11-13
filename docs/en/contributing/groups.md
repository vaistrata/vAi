# Writing custom groups

A group combines `Node` objects and their connections into a reusable component. Create and connect the component `Node` objects inside the group, then specify the inputs and outputs exposed to the outside. Other models or groups can connect to these inputs and outputs to reuse the same configuration.

Groups inherit from `vai::NodeGroup`. As described in the custom node guide, a `Node` is a unit of computation in the graph. A group combines `Node` objects, so the actual computation runs in the `Node` objects inside it. A group can be regarded as a small model built from these `Node` objects. New GPU computations require a new `Node` and kernel; see [Writing custom nodes and kernels](nodes.md).

The main example in this guide is [`FeedForward`](../../../vai/groups/feed-forward.h), used by the `Classic` configuration of `Transformer`. The implementation takes the contributor's configuration, creates two `Linear` `Node` objects, connects them internally, and exposes the group's inputs and outputs. The normalization `Node` objects in [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h) illustrate how to add or omit `Node` objects according to the desired configuration. The layer structure of [`Transformer`](../../../vai/groups/transformer.h) illustrates how to create multiple instances of the same block and connect them in sequence.

## 1. Define the group boundary and inputs and outputs

First decide which `Node` objects to combine into a group. Then define the inputs that supply data to the group and the outputs that return results. Specify a name, shape, and dtype for each input and output.

For example, `FeedForward` combines two `Linear` `Node` objects and an activation computation. Its basic computation is `fc2(activation(fc1(in0)))`. This section starts with the simple configuration `acceptResidual = false`; section 5 covers the additional input.

| Item | `FeedForward` example |
|---|---|
| Input | `in0 [..., inDim]` |
| Intermediate result | `[..., hiddenDim]` |
| Output | `out0 [..., outDim]` |
| dtype | The `Linear` outputs in this example use the engine's default output dtype |
| Computation settings | Input, hidden, and output dimensions, activation function, and whether to use bias |

`...` means that the leading axes are preserved. For example, an input of `[T, inDim]` produces an output of `[T, outDim]`. Specify any dtype and dimension size restrictions according to the conditions supported by the internal `Node` objects.

A group can contain other groups as well as `Node` objects. For example, `TransformerLayer` uses `attention` and `FeedForward` as subgroups. Each subgroup connects its internal `Node` objects, and the parent group connects these subgroups to other components.

## 2. Distinguish configuration values from structural differences

### 2.1 Constructor configuration values

Write the group constructor to receive the required values through `Config`. Specify the meaning and default of each value, and pass it to the `Node` that uses it.

The following `Config` is defined inside `FeedForward`.

```cpp
struct Config {
    uint32_t inDim;
    uint32_t hiddenDim;
    uint32_t outDim;
    vai::Epilogue activation = vai::Epilogue::None;
    bool hasBias = false;
    bool acceptResidual = false;
};
```

`inDim`, `hiddenDim`, and `outDim` determine the sizes of the `Linear` `Node` objects to create.
`activation` is the activation function applied by the first `Linear`. The default, `None`, applies no activation function.
Receive shared values once and pass them to the components that need them, so the group does not define the same value more than once.

Express structural differences according to the following criteria.

| What varies | How to express it | `Transformer` family example |
|---|---|---|
| Dimensions or repetition count | Receive values through `Config`. | `hiddenDim`, `numLayers` |
| Whether a particular `Node` is used | Use `std::optional` and construct it only when needed. | Optional normalization in attention |
| Major differences in member types or connection structure | Consider separate types, for example through templates. | `TransformerType`, `QkvLayout` |

Depth and dimensions are values passed to the constructor. Decide whether to use each `Node` at construction and create only the required members. When component types or connection patterns differ, select the group type at compile time.

### 2.2 Select node and subgroup types

The types of `Node` objects or subgroups can change while keeping the same connection structure. The `Classic` and `Default` configurations of [`TransformerLayer`](../../../vai/groups/transformer-layer.h) share a connection structure but use different normalization `Node` and `FeedForward` group types.

The following type selection code is in the `vai::group` namespace.

```cpp
template<TransformerType layout> struct BranchFfn;
template<> struct BranchFfn<TransformerType::Classic> { using Type = FeedForward; };
template<> struct BranchFfn<TransformerType::Default> { using Type = GatedFeedForward; };
```

The first line declares the `BranchFfn` template. The next two lines specify the definitions for `Classic` and `Default`. Providing separate definitions for specific template arguments is called **template specialization**.

`Type` is `FeedForward` for `Classic` and `GatedFeedForward` for `Default`.
`BranchNorm` selects normalization `Node` types in the same way: `LayerNorm` for `Classic` and `RmsNorm` for `Default`. `TransformerLayer` declares its members with the selected types, so there is no need to write separate constructors and connection code for each configuration.

A model or parent group specifies the layer configuration as follows.

```cpp
using ClassicLayer = vai::group::TransformerLayer<
    vai::TransformerType::Classic, vai::QkvLayout::Separate>;
```

Template arguments can select the type of a `Node` or subgroup. These types are determined at compile time. Settings that differ only in value, such as dimensions or repetition counts, are passed through `Config` when constructing the group.

### 2.3 Specialize groups by connection structure

A group template can be specialized when the number of required `Node` objects or the connection paths differ. Write separate members, constructors, and connection code for each specialization.

The following excerpt shows selected members from the two specializations of [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h).

```cpp
template<QkvLayout layout> struct MultiHeadAttention;

template<>
struct MultiHeadAttention<QkvLayout::Separate> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    vai::node::Linear toQ, toK, toV;
    // ... Remaining members, constructor, and connections
};

template<>
struct MultiHeadAttention<QkvLayout::Fused> : public vai::NodeGroup
{
    using Config = MultiHeadAttentionConfig;

    vai::node::Linear toQKV;
    // ... Remaining members, constructor, and connections
};
```

A model or parent group selects the implementation through a template argument. The selected implementation's constructor creates and connects the `Node` objects for that configuration.

When multiple implementations accept settings with the same meaning, define the configuration struct outside the group and share it. Both specializations in this example use the same configuration type through `using Config = MultiHeadAttentionConfig;`. This allows a model or parent group to pass values to the constructor using the same configuration type even when the internal implementations differ. Settings not shared with other implementations can be defined inside the group, as in `FeedForward::Config`.

## 3. Create components as members

### 3.1 Construct node members and pass configuration

Define the group as a `struct` inheriting from `vai::NodeGroup`. By default, declare the required `Node` objects and subgroups as value members. These members are constructed with the group. Pass configuration to each member through the group constructor's initializer list. The following excerpt shows the members and constructor of `FeedForward`. `FeedForward` is declared as a `struct`, and the initializer list after the constructor's `:` gives each member name and the configuration used to construct it. The example passes `Linear` settings to `fc1` and `fc2`. The constructor body connects these `Node` objects.

```cpp
struct FeedForward : public vai::NodeGroup
{
    vai::node::Linear fc1;
    vai::node::Linear fc2;

    // ... Config from section 2

    FeedForward(Config config)
    : fc1({
        .inDim = config.inDim,
        .outDim = config.hiddenDim,
        .hasBias = config.hasBias,
        .epilogue = config.activation})
    , fc2({
        .inDim = config.hiddenDim,
        .outDim = config.outDim,
        .hasBias = config.hasBias,
        .acceptResidual = config.acceptResidual})
    {
        // ... Connections from section 4 and slot exposure from section 5
    }
};
```

The same `hiddenDim` is passed as the first `Node`'s output dimension and the second `Node`'s input dimension. Configure connected components so their shapes match in this way. The `prepare()` method of the `Node` receiving the actual input also checks the required conditions. Write `Node` configuration so it can be read in the initializer list.

### 3.2 Construct subgroups by type

Declare members with the subgroup types selected in section 2. The following code declares the `ffn` member with the type selected by `TransformerLayer`.

```cpp
typename BranchFfn<layout>::Type ffn;
```

`BranchFfn<layout>::Type` is the subgroup type determined by the selection rules in section 2. `typename` is the C++ keyword that tells the compiler to interpret `Type` as a type name. `ffn` is the name of the member to construct with that type. Use this pattern for the declaration.

Pass configuration matching the selected type when constructing the member. The following constructor excerpt creates `ffn`. Initialization of other members is omitted.

```cpp
TransformerLayer(Config config)
: ffn(ffnConfig(config))
{
    // ... Internal connections and slot exposure
}
```

`ffnConfig(config)` illustrates how to take the parent group's configuration and create the configuration passed to a subgroup. The enclosing `ffn(...)` constructs the `ffn` member with that configuration. A configuration helper like this allows the parent group to share construction code even when each type has a different configuration struct.

The parent group determines the configuration values passed to each subgroup. Each subgroup handles the creation and connection of its own internal `Node` objects.

### 3.3 Conditionally construct `optional` members

`Node` objects and subgroups that are added or omitted according to configuration can be `std::optional` members. Check the configuration in the group constructor and construct each member only when needed. The following excerpt from [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h) creates `Node` objects only when required. It includes the member declaration and code from the constructor body.

```cpp
std::optional<vai::node::RmsNorm> normQ, normK;

// ... Constructor body
if (config.qkNormEps > 0.0f)
{
    normQ.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
    normK.emplace(vai::node::RmsNorm::Config{.eps = config.qkNormEps});
}
```

`emplace()` constructs a `Node` directly inside the `optional` with the supplied configuration. When the condition is not met, the `optional` remains empty. When connecting the graph, the engine's `-` syntax skips an empty `optional` in the middle and directly connects the preceding and following `Node` objects or groups. Section 4 gives a connection example.

Slots reference their owning `Node` and store the addresses of connected slots. Therefore, construct `Node` objects and groups directly in their final storage locations, and do not move or copy them while in use. Do not clear a connected `optional` or reconstruct its `Node` while in use either.

### 3.4 Construct repeated groups and preserve addresses

Repeated blocks can be stored in a vector. `Transformer` has a `std::vector<Layer> layers` member, where `Layer` is an alias for the selected `TransformerLayer` type. The following block construction code is from the `Transformer` constructor. Code that applies per-layer settings is omitted.

```cpp
VAI_ASSERT(config.numLayers >= 1);

layers.reserve(config.numLayers);
for (uint32_t i = 0; i < config.numLayers; ++i)
{
    // ... Calculate settings for this layer
    layers.emplace_back(config.layerCfg);
}
```

Reserve all required storage first with `reserve()`, then construct groups in place with `emplace_back()`. Avoid subsequent operations that move elements or reallocate the vector. Stable addresses are required for the engine to work. For other ownership options, see [Container precautions](../03-usage/node-group.md#4-precautions-when-using-containers).

The parent group determines the repetition count and per-child settings. For example, an option that applies only to the last layer can be set by the parent before constructing that layer.

## 4. Connect internal paths

Connect the `Node` objects and subgroups created in section 3 inside the group constructor body. Each connection defines a path from the output of one `Node` or group to the input of the next.

### 4.1 Connect default inputs and outputs

Both `Node` objects and subgroups use the `-` syntax for connections. Omitting slot names connects `out0` on the left to `in0` on the right. The following is the internal connection in `FeedForward`.

```cpp
fc1 - fc2;
```

As shown above, `fc1`'s `out0` connects directly to `fc2`'s `in0`. Connections to a group use the slots it exposes externally. Section 5 explains how to expose slots. Subgroups of different types can share connection code if they provide slots with the same names and roles.

### 4.2 Connect repeated groups in sequence

Repeated groups in a vector can be connected with a loop that joins each element's output to the next element's input. The following excerpt from [`Transformer`](../../../vai/groups/transformer.h) connects the constructed groups in sequence. Run this loop after constructing all groups in the `layers` vector from section 3.

```cpp
for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
```

Each connection joins `layers[i - 1]`'s `out0` to `layers[i]`'s `in0`.

### 4.3 Select slots by name

To connect a specific slot, specify its name with `/`. A name before a `Node` or group selects an input slot, while a name after it selects an output slot. In the table below, `a` and `b` are `Node` objects or subgroups.

| Connection code | Output slot on the left | Input slot on the right |
|---|---|---|
| `a - b` | `out0` of `a` | `in0` of `b` |
| `a - "in1"/b` | `out0` of `a` | `in1` of `b` |
| `a/"out1" - b` | `out1` of `a` | `in0` of `b` |

Use slot names actually provided by the `Node` or group.

### 4.4 Branch outputs and connect `optional` members

To use one output value in multiple computations, connect the same output slot to each input slot. The following excerpt from [`MultiHeadAttention`](../../../vai/groups/multi-head-attention.h) keeps only the connections that start three paths. Subsequent computations on each path are omitted.

```cpp
input - toQ;
input - toK;
input - toV;
```

`input` is the member name of an `Identity` `Node`, which passes its input value through unchanged. All three lines use `out0` of the same `input` member. Connecting this output to `in0` of `toQ`, `toK`, and `toV` splits it into three paths. The `optional` members from section 3 can appear in the same connection expressions. The following excerpt from the same implementation shows the connections around `normQ`.

```cpp
splitQ - normQ - perQ;
```

If `normQ` contains a `Node`, the connections pass the output of `splitQ` through `normQ` to `perQ`. If it is empty, `out0` of `splitQ` connects directly to `in0` of `perQ`. These connections are determined when wiring the graph in the constructor, so separate connection expressions are not needed for each combination of settings.

### 4.5 Shape and dtype conditions and computation order

Check that the following `Node` or group supports the output shape and dtype of the preceding one. The `-` syntax does not convert shapes or dtypes. If conversion is needed, insert a `Node` that performs it into the path.

Also check computation order when a `Node` option fuses an activation or residual addition. For example, applying activation function `f` and then adding input `r` computes `f(x) + r`. Adding `r` first, as in `f(x + r)`, can change the result. Choose options that preserve the original computation order, or connect a separate `Node` for the required computation.

## 5. Expose slots for external use

After connecting the internal paths in section 4, expose the inputs and outputs for a model or parent group to connect to. In the group constructor, use `defineSlot()` to give an external name to a slot of an internal `Node` or subgroup.

### 5.1 Expose internal slots

The first argument of `defineSlot()` is the name used outside the group. The second is the existing internal slot accessed through that name. This function does not create a new slot or `Tensor`. The following excerpt from the [`FeedForward`](../../../vai/groups/feed-forward.h) constructor exposes its slots.

```cpp
defineSlot("in0", fc1.slot("in0"));
if (config.acceptResidual)
    defineSlot("residual", fc2.slot("residual"));
defineSlot("out0", fc2.slot("out0"));
```

`fc1.slot("in0")` retrieves the existing input slot of `fc1`. `defineSlot("in0", ...)` exposes that slot as the group's `in0`. This example exposes the following slots.

| Group slot name | Internal slot it connects to | Exposure condition |
|---|---|---|
| `in0` | `in0` of `fc1` | Always exposed |
| `out0` | `out0` of `fc2` | Always exposed |
| `residual` | `residual` of `fc2` | `acceptResidual = true` |

Register one internal slot per external name. Registering the same name again does not add a connection target. To split input into multiple paths, connect one internal output to multiple inputs as in section 4.

### 5.2 Expose and connect conditional inputs

Expose a slot added by a setting only when that condition is met. In this example, constructing the group with `acceptResidual = true` adds the `residual` input. Its value is added to the result of `fc2`. The model or parent group must also connect a path that supplies a value to this additional exposed input.

The following example illustrates an external connection. `ffn` is a group constructed with `acceptResidual = true`, and `residualSource` is a `Node` that outputs the value to add.

```cpp
residualSource - "residual"/ffn;
```

With `acceptResidual = false`, this input is not exposed. Specify the input's shape and dtype conditions along with the condition for exposing it. These conditions follow the requirements of the connected internal `Node`. In this example, `residual` uses the same dtype as the group output. See the [`Linear` implementation](../../../vai/nodes/linear.h) for supported shapes.

### 5.3 Expose subgroup slots

To expose a subgroup's input or output, retrieve the slot with `slot()` and pass it to `defineSlot()`. The following code exposes the output of the `ffn` subgroup from section 3 as the parent group's `out0`.

```cpp
defineSlot("out0", ffn.slot("out0"));
```

Connecting to the parent group's `out0` connects to the output of subgroup `ffn`. To use multiple group implementations in the same way, align the names and roles of their exposed slots. The shape and dtype conditions of the connected inputs and outputs must also be compatible.

## 6. Check the build and writing rules

### 6.1 Include group headers and build

Include the group header from the C++ source file or the model or parent group header that uses it. For example, include the following header to use `FeedForward`.

```cpp
#include "vai/groups/feed-forward.h"
```

Building C++ source that includes this header also compiles the required group implementation. A composition of existing `Node` objects and groups does not require a new group-specific kernel. The kernels of the `Node` objects follow their existing build settings. For adding executable targets, see [Writing custom models](models.md#51-registering-an-executable-target); for the build environment and commands, see the [Quickstart](../02-quickstart.md).

### 6.2 File placement and rules for code and comments

| Purpose | File location | Namespace |
|---|---|---|
| Group used by multiple models | `vai/groups/<name>.h` | `vai::group` |
| Group used by a specific model | `models/<model-name>/groups/<name>.h` | The model's namespace |

Stable addresses and correct slot connections are required for the engine to work. Value members, constructors centered on `Config`, and the following code style are repository conventions.

| Item | Writing rule |
|---|---|
| Declaration | Declare as `struct GroupName : public vai::NodeGroup`. |
| Naming | Use `PascalCase` for types and `camelCase` for functions and variables. |
| Initialization | Initialize `Node` configuration with designated field names. Declare required derived values before the members that use them. |
| Shared configuration | When multiple implementations accept the same settings, define a shared type and use `using Config = ...;`. |
| Supported configurations | Make unsupported template choices fail at compile time. |
| Code style | Put function opening braces on the next line. Follow [Common code conventions](../index.md#3-common-code-conventions). |
| Headers | Add header guards and include the headers for the `Node` objects and groups used. Mark regular function definitions outside class definitions as `inline`. |
| Descriptive comments | Describe the computation, inputs and outputs, conditional slots, and supported conditions. |

**Passing external state**

If the group contains a `Node` that needs an external buffer or state, it can provide a method to pass that state to the internal `Node`. Call this method from the model or parent group when needed. For the order of state management calls, see [Model execution methods](models.md#42-managing-state-in-execution-methods).

**Helpers for type-specific configuration**

When each selected component has a different configuration type, a function can use `if constexpr` to create the appropriate configuration. The following is `TransformerLayer::ffnConfig()` with individual field values omitted.

```cpp
static auto ffnConfig(const Config& config)
{
    if constexpr (layout == TransformerType::Classic)
        return FeedForward::Config{/* ... */};
    else
        return GatedFeedForward::Config{/* ... */};
}
```

`if constexpr` compiles only the branch for the selected value. Therefore, `Classic` can return `FeedForward::Config` and `Default` can return `GatedFeedForward::Config`. Keep only configuration logic in this function; component construction and connections belong in the group.

**Supported choices and initialization order**

The shared implementation of `TransformerLayer` uses `static_assert` to accept only `Classic` and `Default`. Other configurations are handled by their respective specializations. As in `MultiHeadAttention`, the primary template can also be declared without a definition, with groups defined only for supported choices. Template specialization and these checks are ways to separate implementations, not formats required by the engine.

Declare and initialize derived values used by multiple members before those members. C++ initializes members in declaration order, not in the order listed in the initializer list.

**Order of configuration changes for repeated blocks**

For callbacks that change per-layer settings, specify the order of application and how long changed values persist. `Transformer`'s `perLayer` modifies the same `config.layerCfg`, so changes persist into the next layer. After the callback, finalize any conditions the parent must guarantee, then construct the child. The following example is an excerpt from the descriptive comment for `FeedForward`.

```cpp
/////////////////////////////////////////////////////////////////////////////////////////
// FeedForward - two-layer MLP, the activation fused into the first projection
// Input: in0 [..., inDim], optional residual [..., outDim]
// Output: out0 [..., outDim] = fc2(activation(fc1(in0)))
//
// ... Optional settings and computation details
/////////////////////////////////////////////////////////////////////////////////////////
```
