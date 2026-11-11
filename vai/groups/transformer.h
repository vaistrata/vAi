#ifndef VAI_GROUPS_TRANSFORMER_H
#define VAI_GROUPS_TRANSFORMER_H

#include "transformer-layer.h"
#include "vai/nodes/layer-norm.h"
#include "vai/nodes/rms-norm.h"
#include <functional>
#include <vector>

namespace vai {


// The tokens the KV cache already holds, which attention and RoPE read. A model
// owns it and rewrites it after sync(), before each forward.
inline eva::Buffer createKvCacheCounter(eva::Device device)
{
    eva::Buffer buffer = device.createBuffer({
        .size = sizeof(uint32_t),
        .usage = eva::BUFFER_USAGE::UNIFORM_BUFFER,
        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT
    });
    *(uint32_t*)buffer.map() = 0;
    return buffer;
}


// Runs on every layer's config, for the fields that vary down the stack; the two
// flags below are applied after it.
using TransformerPerLayer = std::function<void(uint32_t, vai::TransformerLayerConfig&)>;


// The fields of the two configs no set of weights decides, so a checkpoint entry
// leaves them out and the caller stamps them on with TransformerConfig::with().
struct StackOptions {
    bool allowBatch = false;
    uint32_t kvCacheSize = 0;
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    bool sliceLastRow = false;
    bool gatingLastOut = false;
};


// One stack: numLayers copies of layerCfg, whose embedDim and normEps serve the
// whole of it.
struct TransformerConfig {
    uint32_t numLayers;
    bool sliceLastRow = false;
    bool gatingLastOut = false;             // slightly faster when the prefill is chunked
    vai::TransformerLayerConfig layerCfg;
    vai::TransformerPerLayer perLayer;      // absent: every layer is layerCfg

    TransformerConfig with(const vai::StackOptions& opts) const
    {
        TransformerConfig cfg = *this;
        cfg.sliceLastRow = opts.sliceLastRow;
        cfg.gatingLastOut = opts.gatingLastOut;
        cfg.layerCfg.allowBatch = opts.allowBatch;
        cfg.layerCfg.kvCacheSize = opts.kvCacheSize;
        cfg.layerCfg.kvCacheDtype = opts.kvCacheDtype;
        return cfg;
    }
};


}   // namespace vai


namespace vai::group {


// The final norm is the one the layers normalize with.
template<TransformerType type> struct FinalNorm;
template <>
struct FinalNorm<TransformerType::Classic>
{
    using Type = vai::node::LayerNorm;
};
template <>
struct FinalNorm<TransformerType::Default>
{
    using Type = vai::node::RmsNorm;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Transformer - a stack of layers and the norm that closes it
// Input:  in0 [T, embedDim], or [B, T, embedDim] with layerCfg.allowBatch
// Output: out0 the shape of in0, its T 1 under sliceLastRow
/////////////////////////////////////////////////////////////////////////////////////////
template<
    TransformerType type = TransformerType::Default,
    QkvLayout qkv = QkvLayout::Separate
>
struct Transformer : public vai::NodeGroup
{
    using Layer = TransformerLayer<type, qkv>;

    std::vector<Layer> layers;
    typename FinalNorm<type>::Type finalNorm;

    using Config = TransformerConfig;

    Transformer(Config config) : finalNorm({.eps = config.layerCfg.normEps})
    {
        VAI_ASSERT(config.numLayers >= 1);

        layers.reserve(config.numLayers);
        for (uint32_t i = 0; i < config.numLayers; ++i)
        {
            const bool last = i == config.numLayers - 1;
            if (config.perLayer)
                config.perLayer(i, config.layerCfg);
            config.layerCfg.sliceLastRow = config.sliceLastRow && last;
            config.layerCfg.gatingLastOut = config.gatingLastOut && last;
            layers.emplace_back(config.layerCfg);
        }

        for (size_t i = 1; i < layers.size(); ++i)
            layers[i - 1] - layers[i];
        layers.back() - finalNorm;

        defineSlot("in0", layers[0].slot("in0"));
        defineSlot("out0", finalNorm.slot("out0"));
    }

    void setCausalMask(eva::Buffer buffer)
    {
        for (auto& layer : layers)
            layer.mhAttn.attn.setCausalMask(buffer);
    }

    void setKvCacheCounter(eva::Buffer buffer)
    {
        for (auto& layer : layers)
            layer.mhAttn.setKvCacheCounter(buffer);
    }

    // Closing it drops the last layer's FFN and everything past it, finalNorm included.
    void setGateOpen(bool open)
    {
        auto& gate = layers.back().gate;
        if (gate) gate->setOpen(open);
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_TRANSFORMER_H
