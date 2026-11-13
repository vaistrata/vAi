# NodeGroup

A `NodeGroup` is a subgraph that wraps multiple nodes so they can be used as one unit. It groups repeated structures, such as transformer blocks.

First, check [`vai/groups`](../../../vai/groups/) for a reusable group.

```cpp
struct FeedForward : public vai::NodeGroup {
    vai::node::Linear fc1, fc2;                 // 1

    // ... Config definition

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
        fc1 - fc2;                            // 2

        defineSlot("in0", fc1.slot("in0"));   // 3
        if (config.acceptResidual)
            defineSlot("residual", fc2.slot("residual"));
        defineSlot("out0", fc2.slot("out0"));
    }
};
```

1. Declare nodes as members,
2. connect the internal nodes in the constructor, and
3. choose which slots to expose outside the group.

`NodeGroup::defineSlot` assigns an external name to an internal `Node` slot rather than creating a new slot. A connection to the group's `in0` connects to the `in0` slot of `fc1`.

`NodeGroup` has neither `prepare()` nor `run()`. The `Node` instances inside it execute. `NodeGroup` combines multiple `Node` instances into a subgraph so repeated structures can be reused.

## 1. Connections

Connect groups with `-`, just like `Node`. `NodeGroup` and `Node` can be mixed.

```cpp
norm - ffn - head;          // Node - group - node
ffn1 - ffn2;                // Group - group
```

The `/` slot selection rules are the same. Before a group, it selects an input slot; after a group, it selects an output slot. Only names exposed with `defineSlot` are available. As with `Node`, omitting `/` selects `in0` and `out0`.

## 2. Nesting

A `NodeGroup` can contain other `NodeGroup` instances. Llama 3.2's TransformerLayer is a `NodeGroup` with `MultiHeadAttention` and `GatedFeedForward` members. Both members are also `NodeGroup` instances made up of multiple `Node` instances.

```cpp
template<TransformerType layout, QkvLayout qkv>
struct TransformerLayer : public vai::NodeGroup {
    vai::node::Identity input;
    typename BranchNorm<layout>::Type norm1;
    MultiHeadAttention<qkv> mhAttn;
    std::optional<vai::node::Slice> lastRow;
    std::optional<vai::node::Gate> gate;
    typename BranchNorm<layout>::Type norm2;
    typename BranchFfn<layout>::Type ffn;

    TransformerLayer(...)
    {
        input - norm1 -            mhAttn - lastRow - gate/"then" - norm2 -            ffn;
        input         - "residual"/mhAttn - lastRow - gate/"then"         - "residual"/ffn;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", ffn.slot("out0"));
    }
};
```

In Llama 3, `norm1` and `norm2` are `RmsNorm`, and `ffn` is `GatedFeedForward`. The first line is the main path through normalization, Attention, and the FFN; the second line is the residual path.

## 3. Identity

In the example above, `input` is an `Identity` that performs no computation. It provides a branching point when input splits into multiple paths. It is used when the same input must reach two destinations, as in a residual connection.

## 4. Precautions when using containers

Both `Node` and `NodeGroup` can be stored in containers. However, slots connected with `-` hold each other's addresses, so the elements must keep their addresses without being copied or moved.

```cpp
layers.reserve(config.numLayers);
for (uint32_t i = 0; i < config.numLayers; ++i)
{
    // ... Per-layer configuration
    layers.emplace_back(config.layerCfg);
}

for (size_t i = 1; i < layers.size(); ++i)
    layers[i - 1] - layers[i];
```

When a `std::vector` runs out of capacity, it moves existing elements to new memory. Inserting a temporary node with `push_back(Layer(...))` also causes a copy or move. Slot parent references or connections may still point to the original objects, causing access to the wrong objects during execution. Compilation succeeds, but graph execution fails.

The author must keep elements at stable addresses. There are two approaches.

**1. `reserve` + `emplace_back`**: Reserve enough space to prevent reallocation, then construct elements directly in the container to avoid copies. The example above uses this approach.

**2. `std::unique_ptr`**: Allocate elements on the heap and store only pointers in the container. Moving pointers within the vector does not move the objects they point to.

```cpp
std::vector<std::unique_ptr<Layer>> layers;
layers.push_back(std::make_unique<Layer>(layerCfg));
```

vAi allows the second approach but recommends the first.

Before adding a group, check whether an existing group can be reused.

Elements in other STL containers, such as `std::deque`, `std::list`, and `std::map`, must also remain where they were constructed. Which operations move elements depends on the container; those standard C++ rules are outside the scope of this document.
