# NeuralNet

`NeuralNet` is the entire graph. The way to create it is almost the same as with the [NodeGroup](node-group.md). Declare nodes as members and connect them with `-` in the constructor.

The difference is how to define the input and output of the graph. NodeGroup puts names on inside slots via `NodeGroup::defineSlot()` and exposes them outside, while `NeuralNet` connects directly to the boundary through `NeuralNet::input()` and `NeuralNet::output()`.

```cpp
struct Block : public vai::NodeGroup {       // NodeGroup
    defineSlot("in0", fc1.slot("in0"));      // Naming an inside slot and exposing it outside
};

struct Net : public vai::NeuralNet {       // Neural Network
    input("in0") - fc1;                      // Directly connecting to the input boundary
};
```

`input()` and `output()` are the only points where the graph meets the outside. Only those connected here can put values into or get values from the host.

## Creation

```cpp
struct Net : public vai::NeuralNet {
    vai::node::TokenEmbedding embedding;
    std::vector<TransformerBlock> layers;
    vai::node::Linear lmHead;

    Net(const vai::NeuralNetConfig& netCfg, ModelConfig modelCfg)
    : NeuralNet(netCfg)
    , embedding({ /* ... */ })
    , lmHead({ modelCfg.embedDim, modelCfg.vocabSize })
    {
        layers.reserve(modelCfg.numLayers);
        for (uint32_t i = 0; i < modelCfg.numLayers; ++i)
            layers.emplace_back(/* ... */);

        input("inTokenIds") - embedding - layers[0];

        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];

        layers.back() - lmHead - output("nextTokenId");
    }
};
```

Using containers like `std::vector` follows the same rules as [Precautions when using containers in NodeGroup](node-group.md#precautions-when-using-containers).

## Boundary name

`input()` and `output()` are distinguished by name, if omitted, they are `"in0"` and `"out0"` respectively.

```cpp
input()              // == input("in0")
input("inTokenIds")  // Making an input boundary named "inTokenIds"
```

The boundary is created at the first call. You don't need to declare it in advance, and if you call the same name again, the same boundary will be returned. If there are multiple inputs or outputs, you can give different names to them.

The names set here are the keys used during execution.

## Execution

Pass named input tensors and pointers to the output tensors to `NeuralNet::operator()`.

```cpp
using TensorDict = std::map<std::string, Tensor>;
using TensorRefDict = std::map<std::string, Tensor*>;

void NeuralNet::operator()(TensorDict inputs, TensorRefDict outputs, bool syncPrev = true);
```

The key for both input and output is the boundary name. The keys of the `TensorDict` passed correspond to the input boundaries created by `input()`, and the keys of the output `TensorRefDict` correspond to the output boundaries created by `output()`.

```cpp
namespace gpt2 = model::production::gpt2;

gpt2::Net net({}, gpt2::ModelConfig::_124M());

Tensor tokens = Tensor(/* ... */);

vai::TensorDict in = {{"inTokenIds", tokens}};
Tensor result;
net(in, {{"nextTokenId", &result}});
```

The input dictionary can also be constructed at the call site.

```cpp
net({{"inTokenIds", tokens}}, {{"nextTokenId", &result}});
```

The call is **asynchronous**. The fact that `model(...)` has returned does not mean that the GPU calculation is complete. On the result tensor, if needed at the time `Tensor::downloadToHost()` or `Tensor::getData<T>()` is called, it waits for completion and then gets the data.
(see [Reading output in Tensor](tensor.md#reading-output)).

The internal process of a single call is covered step by step in the [Execution Pipeline](execution-pipeline.md).

The name passed must exactly match the boundary created by `input()`. An empty input dictionary or an unknown name causes an error. Omitting one of several inputs prunes the branch that starts at that input.

```cpp
net({{"inTokenIds", tokens}}, {{"nextTokenId", &result}});                  // OK
net({}, {{"nextTokenId", &result}});                                        // X: Input is missing.
net({{"inTokenIds", t}, {"extra", e}}, {{"nextTokenId", &result}});         // X: Non-existent boundary
```

## `onForwardBegin`

The basic path that delivers values to the GPU is the input tensor of `NeuralNet::operator()`. However, there are values that do not flow along the graph but need to be changed for each execution.

* The number of tokens processed by the KV cache so far: Attention decides where to continue writing to the cache.
* Random number to be used for this sampling: It is used when `TopKSampling` picks a token using the inverse CDF.

It is not data that flows between nodes, like tokens or logits, but a small state value that is referenced when a specific kernel is executed.

### Buffer setup

Store it in a separate buffer that the host can directly write to, and make the necessary nodes point to that buffer.

```cpp
kvCacheCounter = device().createBuffer({
    .size = sizeof(uint32_t),
    .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
    .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE |
                   eva::MEMORY_PROPERTY::HOST_COHERENT
});

embedding.setKvCacheCounter(kvCacheCounter);

for (auto& layer : layers)
    layer.mhAttn.attn.setKvCacheCounter(kvCacheCounter);
```

As it is `HOST_VISIBLE`, the CPU can directly write the value using `map()`. Connect this buffer to the node once, and change only the values inside the buffer for each execution.

This requires synchronization. If the CPU overwrites the value while the GPU is reading the buffer, they may read different values during the same execution. In addition, as `NeuralNet::operator()` returns without waiting for the GPU, the end of the call does not mean that the GPU execution is over.

### Updating values

You should not change the values at any time. The update should be done after the previous execution is over and before the current execution starts. That point is `NeuralNet::onForwardBegin()`.

```cpp
void onForwardBegin(const vai::TensorDict& inputs) override
{
    // The number of tokens processed before the current execution.
    *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;

    tokensSoFar_ += (uint32_t)inputs.at("inTokenIds").shape()[0];

    topkSampling.setUniform(
        (float)std::rand() / ((float)RAND_MAX + 1.0f));
}
```

In the execution order, the position of the hook is as follows.

```text
Waiting for the previous execution to complete
  → onForwardBegin()        ← Values are updated here
  → Prepare for the current execution
  → Submit the command buffer to the GPU
```

Because of this point, two things are guaranteed.

* **Before the Hook**: You can modify the buffer without conflicting with the GPU, as the previous execution is already complete.
* **After the Hook**: The preparation and submission of the current execution haven't started yet, so the value changed here will be reflected in the current execution.

### Why this is done

These values can be put directly into the GPU command like push constants. This is similar to the way kernel arguments are passed in CUDA. However, if the values recorded in the command change, the command buffer needs to be re-recorded.

Conversely, if the command is set to point to the buffer and only the actual value in the buffer is changed, the command buffer itself can be reused. The things that change for each execution are only a few bytes in the buffer.

This is the key role of `onForwardBegin()`. Without re-recording the command buffer, new values can be delivered to the GPU every time.

If you separate the values that change every time from the command, you don't need to re-record the command buffer even if the KV cache counter or sampling random number keeps changing. This is the part that enables [High performance in Overview](../index.md#3-high-performance).

When the command buffer is actually re-recorded is covered in [5. Re-recording assessment of the Execution pipeline](execution-pipeline.md#5-re-recording-assessment).

### When modifying the same value outside the execution

There are times when you need to modify the same buffer outside `onForwardBegin()`, like when clearing the KV cache. As the previous execution might still be reading that value, wait for completion via `NeuralNet::syncPrevForward()` first.

```cpp
void resetCache()
{
    syncPrevForward();      // The GPU might still be reading the counter.

    tokensSoFar_ = 0;
    *(uint32_t*)kvCacheCounter.map() = 0;
}
```

You can safely modify the values as the GPU no longer uses that buffer afterward.
