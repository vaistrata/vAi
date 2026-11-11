/*=============================================================================
  Written by   oneway@vaistrata.com,
               qkrtmddn0507@vaistrata.com
  Model        Gemma 3 (270M / 1B / 4B(VLM) / 12B(VLM))
=============================================================================*/

#ifndef PRODUCTION_GEMMA3_NET_H
#define PRODUCTION_GEMMA3_NET_H

#define TEST_INDEX_COPY

#include "models/config-base.h"
#include "groups/vision-tower.h"
#include "groups/transformer.h"
#include "vai/nodes/linear.h"
#ifdef TEST_INDEX_COPY
#include "vai/nodes/reshape.h"
#include "vai/nodes/index-copy.h"
#else
#include "vai/nodes/row-slice-scatter.h"
#endif
#include "vai/nodes/token-embedding.h"
#include "vai/nodes/topk-sampling.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string_view>


namespace model::production::gemma3 {


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Gemma 3 (270M / 1B: text only, 4B / 12B: image + text)
/////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t embedDim(std::string_view variant)
{
    if (variant == "270M")
        return 640;
    if (variant == "1B")
        return 1152;
    if (variant == "4B")
        return 2560;
    if (variant == "12B")
        return 3840;
    throw std::invalid_argument("unknown Gemma 3 variant");
}

static constexpr uint32_t numQHeads(std::string_view variant)
{
    if (variant == "270M")
        return 4;
    if (variant == "1B")
        return 4;
    if (variant == "4B")
        return 8;
    if (variant == "12B")
        return 16;
    throw std::invalid_argument("unknown Gemma 3 variant");
}

static constexpr uint32_t numKVHeads(std::string_view variant)
{
    if (variant == "270M")
        return 1;
    if (variant == "1B")
        return 1;
    if (variant == "4B")
        return 4;
    if (variant == "12B")
        return 8;
    throw std::invalid_argument("unknown Gemma 3 variant");
}

// Local layers: plain RoPE at 10K. Every sixth (global): 1M, linearly scaled by 8 on the VLMs.
static constexpr vai::RopeConfig rope(std::string_view variant, uint32_t layer)
{
    if ((layer + 1) % 6 != 0)
        return {.type = vai::RopeType::Default, .base = 10000.0f};
    if (variant == "270M" || variant == "1B")
        return {.type = vai::RopeType::Default, .base = 1000000.0f};
    if (variant == "4B" || variant == "12B")
        return {.type = vai::RopeType::Linear, .base = 1000000.0f, .factor = 8.0f};
    throw std::invalid_argument("unknown Gemma 3 variant");
}


struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/
    LanguageModelConfig languageCfg;
    uint32_t slidingWindow;   // a local layer's span; the global ones attend the whole prefix
    vai::RopeConfig globalRope;
    vai::RopeConfig localRope;
    std::optional<group::VisionTowerConfig> visionCfg;

    group::TransformerLayer::Config decodeBlock = {
        .embedDim = embedDim(variant),
        .headDim = 256,
        .numQHeads = numQHeads(variant),
        .numKVHeads = numKVHeads(variant),
        // .ffnHiddenDim = ffnHiddenDim(variant),
        .normEps = 1e-6f,
    };

    uint32_t headDim = 256;
    bool withQkNorm = true;
    vai::Epilogue ffnActivation = vai::Epilogue::GeluTanh;
    float normEps = 1e-6f;

    static bool isGlobalLayer(uint32_t layer)
    {
        static constexpr uint32_t globalLayerPeriod = 6;
        return (layer + 1) % globalLayerPeriod == 0;
    }

    static ModelConfig get(std::string_view variant)
    {
        return {
            .variant = variant.data(),
            .decodeBlock =
                {
                    .embedDim = embedDim(variant),
                    .headDim = 256,
                    .numQHeads = numQHeads(variant),
                    .numKVHeads = numKVHeads(variant),
                    .normEps = 1e-6f,
                },
        };
    }

    static ModelConfig _270M()
    {
        return {.variant = "270M",
                .languageCfg =
                    {
                        vai::TransformerConfig{.numLayers = 18,
                                               .layerCfg = {.embedDim = 640,
                                                            .headDim = 256,
                                                            .numQHeads = 4,
                                                            .numKVHeads = 1,
                                                            .withQkNorm = true,
                                                            .ffnHiddenDim = 2048,
                                                            .ffnActivation = vai::Epilogue::GeluTanh,
                                                            .normEps = 1e-6f}},
                        262144,   // vocabSize
                        32768     // maxTrainedContext
                    },
                .slidingWindow = 512,
                .globalRope = rope(1000000.0f),
                .localRope = rope(10000.0f)};
    }

    static ModelConfig _1B()
    {
        return {.variant = "1B",
                .languageCfg =
                    {
                        vai::TransformerConfig{.numLayers = 26,
                                               .layerCfg = {.embedDim = 1152,
                                                            .headDim = 256,
                                                            .numQHeads = 4,
                                                            .numKVHeads = 1,
                                                            .withQkNorm = true,
                                                            .ffnHiddenDim = 6912,
                                                            .ffnActivation = vai::Epilogue::GeluTanh,
                                                            .normEps = 1e-6f}},
                        262144,   // vocabSize
                        32768     // maxTrainedContext
                    },
                .slidingWindow = 512,
                .globalRope = rope(1000000.0f),
                .localRope = rope(10000.0f)};
    }

    static ModelConfig _4B()
    {
        const uint32_t textEmbedDim = 2560;

        return {
            .variant = "4B",
            .languageCfg =
                {
                    vai::TransformerConfig{.numLayers = 34,
                                           .layerCfg = {.embedDim = textEmbedDim,
                                                        .headDim = 256,
                                                        .numQHeads = 8,
                                                        .numKVHeads = 4,
                                                        .withQkNorm = true,
                                                        .ffnHiddenDim = 10240,
                                                        .ffnActivation = vai::Epilogue::GeluTanh,
                                                        .normEps = 1e-6f}},
                    262208,   // vocabSize
                    131072    // maxTrainedContext
                },
            .slidingWindow = 1024,
            .globalRope = linearRope(1000000.0f),
            .localRope = rope(10000.0f),
            .visionCfg = group::VisionTowerConfig(textEmbedDim),
        };
    }

    static ModelConfig _12B()
    {
        const uint32_t textEmbedDim = 3840;

        return {
            .variant = "12B",
            .languageCfg =
                {
                    vai::TransformerConfig{.numLayers = 48,
                                           .layerCfg = {.embedDim = textEmbedDim,
                                                        .headDim = 256,
                                                        .numQHeads = 16,
                                                        .numKVHeads = 8,
                                                        .withQkNorm = true,
                                                        .ffnHiddenDim = 15360,
                                                        .ffnActivation = vai::Epilogue::GeluTanh,
                                                        .normEps = 1e-6f}},
                    262208,   // vocabSize
                    131072    // maxTrainedContext
                },
            .slidingWindow = 1024,
            .globalRope = linearRope(1000000.0f),
            .localRope = rope(10000.0f),
            .visionCfg = group::VisionTowerConfig(textEmbedDim),
        };
    }

private:
    // Plain RoPE, so the two bands differ in the only field it reads.
    static vai::RopeConfig rope(float base) { return {.type = vai::RopeType::Default, .base = base}; }

    static vai::RopeConfig linearRope(float base)
    {
        return {.type = vai::RopeType::Linear, .base = base, .factor = 8.0f};
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
    uint32_t prefillChunk = 512;   // 0: the whole prompt in one forward; an image never splits
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;

    std::optional<group::VisionTower> visionTower;
#ifdef TEST_INDEX_COPY
    vai::node::Reshape imageRows;
    vai::node::IndexCopy imageCopy;
#else
    vai::node::RowSliceScatter imageScatter;
#endif

    vai::node::TokenEmbedding embedding;
    std::vector<group::TransformerLayer> decoder;   // every sixth global, the rest local (a ring cache)
    vai::node::RmsNorm decoderNorm;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;
    eva::Buffer localMask;   // causalMask spanning the sliding window, the local layers'
#ifdef TEST_INDEX_COPY
    eva::Buffer imageIndex;
#endif

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
        : NeuralNet(netCfg), modelCfg(modelCfg), runtimeCfg(runtimeCfg),
          maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.languageCfg.maxTrainedContext))
#ifdef TEST_INDEX_COPY
          ,
          imageRows({-1, int64_t(modelCfg.languageCfg.layerCfg.embedDim)})
#else
          ,
          imageScatter({.sliceSize = modelCfg.visionCfg ? modelCfg.visionCfg->tokensPerImage : 0})
#endif
          ,
          embedding({.vocabSize = modelCfg.languageCfg.vocabSize,
                     .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
                     .scale = std::sqrt(float(modelCfg.languageCfg.layerCfg.embedDim))}),
          decoderNorm({.eps = modelCfg.languageCfg.layerCfg.normEps}),
          lmHead({.inDim = modelCfg.languageCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.vocabSize})
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        if (modelCfg.visionCfg)
        {
            // A chunk holds its images whole.
            VAI_ASSERT(runtimeCfg.prefillChunk == 0 || runtimeCfg.prefillChunk >= modelCfg.visionCfg->tokensPerImage);
            visionTower.emplace(*modelCfg.visionCfg);
        }

        const auto& layerCfg = modelCfg.languageCfg.layerCfg;
        const uint32_t numLayers = modelCfg.languageCfg.numLayers;
        decoder.reserve(numLayers);
        for (uint32_t i = 0; i < numLayers; ++i)
        {
            const bool global = ModelConfig::isGlobalLayer(i);
            const bool last = i == numLayers - 1;
            decoder.emplace_back(group::TransformerLayer::Config{
                .embedDim = layerCfg.embedDim,
                .headDim = layerCfg.headDim,
                .numQHeads = layerCfg.numQHeads,
                .numKVHeads = layerCfg.numKVHeads,
                .ffnHiddenDim = layerCfg.ffnHiddenDim,
                .normEps = layerCfg.normEps,
                .rope = global ? modelCfg.globalRope : modelCfg.localRope,
                .kvCacheSize = global ? maxRuntimeContext : localCacheSize(modelCfg, runtimeCfg, maxRuntimeContext),
                .kvCacheDtype = runtimeCfg.kvCacheDtype,
                .kvCacheRing = !global,
                .sliceLastRow = last,
                .gatingLastOut = last && runtimeCfg.prefillChunk > 0});
        }

#ifdef TEST_INDEX_COPY
        input("inTokenIds") - embedding - "in0" / imageCopy;
        input("inImages") - visionTower - imageRows - "in1" / imageCopy;
        imageCopy - decoder[0];
#else
        input("inTokenIds") - embedding - "in0" / imageScatter;
        input("inImages") - visionTower - "in1" / imageScatter;
        imageScatter - decoder[0];
#endif
        for (uint32_t i = 1; i < numLayers; ++i)
            decoder[i - 1] - decoder[i];
        decoder.back() - decoderNorm - lmHead - topkSampling - output("nextTokenId");

        kvCacheCounter = vai::createKvCacheCounter(device());
        causalMask = vai::createCausalMask(device(), 1);
        localMask = vai::createCausalMask(device(), 1, {}, modelCfg.slidingWindow);
        for (auto& layer : decoder)
        {
            layer.mhAttn.setKvCacheCounter(kvCacheCounter);
            layer.mhAttn.attn.setCausalMask(layer.mhAttn.attn.config_.kvCacheRing ? localMask : causalMask);
        }

        if (visionTower)
        {
#ifdef TEST_INDEX_COPY
            imageIndex = device().createBuffer(
                {.size = MAX_CAUSAL_BLOCKS * modelCfg.visionCfg->tokensPerImage * sizeof(uint32_t),
                 .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                 .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT});
            imageCopy.setIndex(imageIndex);
#endif
        }
    }

    uint32_t tokensSoFar_ = 0;

    // Input: images [B, C, S, S] (or [C, S, S]); empty for text only. promptTokenIds [T].
    // Output: nextTokenId [1], the first generated token.
    // Images must match placeholder blocks in prompt order.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);

        const uint32_t total = promptTokenIds.shape()[0];
        const auto& shape = images.shape();
        const uint32_t imageCount = !images.validShape() ? 0 : shape.size() == 4 ? shape[0] : 1;

        if (imageCount > 0)
        {
            if (!modelCfg.visionCfg || !visionTower)
                throw std::runtime_error("image input is not supported by this Gemma 3 checkpoint");
            const auto& cfg = *modelCfg.visionCfg;
            VAI_ASSERT(shape.size() == 3 || shape.size() == 4);
            VAI_ASSERT(shape[shape.size() - 3] == cfg.imageChannels);
            VAI_ASSERT(shape[shape.size() - 2] == cfg.imageSize);
            VAI_ASSERT(shape.back() == cfg.imageSize);
            if (!promptTokenIds.hasHostData())
                throw std::runtime_error("image prefill requires host-visible prompt tokens");
        }

        // Image blocks use bidirectional attention; text blocks are causal.
        struct Block
        {
            uint32_t end;
            bool causal;
        };
        std::vector<Block> blocks;
        std::vector<uint32_t> imageStarts;
        const uint32_t tokensPerImage = modelCfg.visionCfg ? modelCfg.visionCfg->tokensPerImage : 0;
        {
            const uint32_t imageTokenId = modelCfg.visionCfg ? modelCfg.visionCfg->imageTokenId : 0;
            const uint32_t* ids =
                modelCfg.visionCfg ? static_cast<const uint32_t*>(promptTokenIds.hostData()) : nullptr;
            uint32_t last = 0;

            for (uint32_t pos = 0; ids && pos < total;)
            {
                if (ids[pos] != imageTokenId)
                {
                    ++pos;
                    continue;
                }

                if (imageCount <= imageStarts.size() || total - pos < tokensPerImage ||
                    !std::all_of(
                        ids + pos, ids + pos + tokensPerImage, [&](uint32_t id) { return id == imageTokenId; }))
                    throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

                if (last < pos)
                    blocks.push_back({pos, true});

                imageStarts.push_back(pos);
                pos += tokensPerImage;
                blocks.push_back({pos, false});
                last = pos;
            }

            if (imageStarts.size() != imageCount)
                throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

            if (last < total)
                blocks.push_back({total, true});
        }

        const uint32_t chunk = runtimeCfg.prefillChunk ? runtimeCfg.prefillChunk : total;

        // Reuse the buffer: a single chunk may have no host data after upload.
        vai::Tensor chunkInput = promptTokenIds;

        for (uint32_t off = 0, image = 0; off < total;)
        {
            // The chunk ends before an image it would split.
            uint32_t end = std::min(off + chunk, total);
            for (uint32_t start : imageStarts)
                if (start < end && end < start + tokensPerImage)
                    end = start;
            const uint32_t n = end - off;
            const bool producesToken = (end == total);

            if (n != total)
            {
                VAI_ASSERT(promptTokenIds.hasHostData());

                if (chunkInput.shape()[0] != n)
                    chunkInput = vai::Tensor({n});
                chunkInput.setData((const uint32_t*)promptTokenIds.hostData() + off);
            }

            // The blocks the chunk reaches, their ends counted from its start.
            vai::CausalBlockEnd blockEnd = {};
            uint32_t causalBits = 0, blockCount = 0;
            for (const Block& b : blocks)
            {
                if (b.end <= off)
                    continue;
                if (MAX_CAUSAL_BLOCKS - 1 < blockCount || (MAX_CAUSAL_BLOCKS - 2 < blockCount && b.end < end))
                    throw std::runtime_error("too many Gemma 3 image/text attention blocks");

                causalBits = (causalBits << 1) | uint32_t(b.causal);
                if (b.end < end)
                    blockEnd[blockCount] = b.end - off;
                ++blockCount;
                if (end <= b.end)
                    break;
            }

            // The images the chunk holds, sliced out of the batch unless it is all of it.
            std::vector<uint32_t> chunkStarts;
            const uint32_t firstImage = image;
            for (; image < imageCount && imageStarts[image] < end; ++image)
                chunkStarts.push_back(imageStarts[image] - off);
            const uint32_t chunkImageCount = image - firstImage;

            vai::Tensor chunkImages;
            if (chunkImageCount == imageCount)
                chunkImages = images;
            else if (chunkImageCount > 0)
            {
                VAI_ASSERT(images.hasHostData() && images.dtype() == vai::Dtype::Fp32);
                const size_t imageFloats = images.numElements() / imageCount;
                chunkImages = vai::Tensor({chunkImageCount, shape[1], shape[2], shape[3]});
                chunkImages.setData((const float*)images.hostData() + firstImage * imageFloats);
            }

            sync();   // The GPU may still be reading the masks and imageIndex.
            vai::writeCausalMask(causalMask, causalBits, blockEnd);
            vai::writeCausalMask(localMask, causalBits, blockEnd, modelCfg.slidingWindow);

#ifdef TEST_INDEX_COPY
            uint32_t* rows = imageIndex ? (uint32_t*)imageIndex.map() : nullptr;
            for (uint32_t start : chunkStarts)
                for (uint32_t t = 0; t < tokensPerImage; ++t)
                    *rows++ = start + t;
#else
            imageScatter.setRowStarts(std::move(chunkStarts));
#endif

            step(chunkImages, chunkInput, producesToken ? nextTokenId : nullptr);
            off = end;
        }
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);
        step({}, *inoutTokenId, inoutTokenId);
    }

    void resetCache()
    {
        sync();   // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }

private:
    // A local layer's ring: the window and a chunk past it, padded to the 256
    // the ring wants, or the whole context when that is less.
    static uint32_t localCacheSize(const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg, uint32_t context)
    {
        const uint32_t chunk = runtimeCfg.prefillChunk ? runtimeCfg.prefillChunk : context;
        const uint32_t size = std::min(modelCfg.slidingWindow + chunk - 1, context);
        return (size + 255) / 256 * 256;
    }

    void step(const vai::Tensor& inImages, const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();   // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        // Closing it drops the last layer's FFN and everything past it.
        if (auto& gate = decoder.back().gate)
            gate->setOpen(nextTokenId != nullptr);

        if (nextTokenId)
            topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs;
        if (nextTokenId)
            outputs.emplace("nextTokenId", nextTokenId);

        vai::TensorDict inputs = {{"inTokenIds", inTokenIds}};
        if (inImages.validShape())
            inputs.emplace("inImages", inImages);

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::production::gemma3

#endif
