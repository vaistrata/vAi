/*=============================================================================
  Written by   oneway@vaistrata.com,
               qkrtmddn0507@vaistrata.com
  Model        Gemma 3 (270M / 1B / 4B(VLM) / 12B(VLM))
=============================================================================*/

#ifndef PRODUCTION_GEMMA3_NET_H
#define PRODUCTION_GEMMA3_NET_H

#include "groups/vision-encoder.h"
#include "groups/transformer.h"
#include "vai/nodes/index-copy.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/token-embedding.h"
#include "vai/nodes/topk-sampling.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string_view>


namespace model::production::gemma3 {


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Gemma 3 (270M / 1B: text only, 4B / 12B: image + text)
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/
    TransformerConfig trf;
    uint32_t vocabSize;
    uint32_t maxTrainedContext;

    static constexpr uint32_t imageTokenId = 262144;   // <image_soft_token>, the placeholder an image's tokens replace

    constexpr bool hasVision() const
    {
        const std::string_view name{ variant };
        return name == "4B" || name == "12B";
    }

    static constexpr ModelConfig get(std::string_view variant)
    {
        constexpr ModelConfig variants[] = {
            //  variant  text {blocks  D  Hq  Hkv  Dff  window  ropeFactor}  vocab  maxContext
            { "270M", { 18, 640, 4, 1, 2048, 512, 1.0f }, 262144, 32768 },
            { "1B", { 26, 1152, 4, 1, 6912, 512, 1.0f }, 262144, 32768 },
            { "4B", { 34, 2560, 8, 4, 10240, 1024, 8.0f }, 262208, 131072 },
            { "12B", { 48, 3840, 16, 8, 15360, 1024, 8.0f }, 262208, 131072 },
        };
        for (const ModelConfig& v : variants)
            if (variant == v.variant)
                return v;
        throw std::invalid_argument("unknown Gemma 3 variant");
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
    uint32_t prefillChunk = 512;   // 0: the sliding window; an image never splits
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;
    uint32_t prefillChunk;

    std::optional<group::VisionEncoder> visionEncoder;
    vai::node::IndexCopy imageCopy;

    vai::node::TokenEmbedding embedding;
    group::Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;
    eva::Buffer localMask;   // causalMask spanning the sliding window, the local blocks'
    eva::Buffer imageIndex;   // the image tokens' rows in this chunk

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.maxTrainedContext))
    , prefillChunk(runtimeCfg.prefillChunk ? runtimeCfg.prefillChunk : modelCfg.trf.slidingWindow)
    , visionEncoder(
          modelCfg.hasVision()
              ? std::make_optional<group::VisionEncoder>(
                    group::VisionEncoder::Config{ .base = {}, .outDim = modelCfg.trf.embedDim })
              : std::nullopt)
    , embedding({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.trf.embedDim,
          .scale = std::sqrt(float(modelCfg.trf.embedDim)),
      })
    , decoder({
          .base = modelCfg.trf,
          .kvCacheSize = maxRuntimeContext,
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .maxForwardTokens = prefillChunk,
      })
    , lmHead({ .inDim = modelCfg.trf.embedDim, .outDim = modelCfg.vocabSize })
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;
        this->runtimeCfg.prefillChunk = prefillChunk;

        // A chunk holds its images whole.
        VAI_ASSERT(!visionEncoder || prefillChunk >= visionEncoder->tokensPerImage);

        // clang-format off
        input("inTokenIds") - embedding -   "in0"/imageCopy;
        input("inImages") - visionEncoder - "in1"/imageCopy;
        imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        localMask = vai::createCausalMask(device(), 1, {}, modelCfg.trf.slidingWindow);
        decoder.setCausalMask(causalMask, localMask);

        if (visionEncoder)
        {
            imageIndex = device().createBuffer({
                .size = MAX_CAUSAL_BLOCKS * visionEncoder->tokensPerImage * sizeof(uint32_t),
                .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
            });
            imageCopy.setIndex(imageIndex);
        }
    }

    uint32_t tokensSoFar_ = 0;

    // Input: images [B, C, S, S]; empty for text only. promptTokenIds [T].
    // Output: nextTokenId [1], the first generated token.
    // Images must match placeholder blocks in prompt order.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);

        const uint32_t total = promptTokenIds.shape()[0];
        const auto& shape = images.shape();
        const uint32_t imageCount = images.validShape() ? shape[0] : 0;

        if (imageCount > 0)
        {
            if (!modelCfg.hasVision() || !visionEncoder)
                throw std::runtime_error("image input is not supported by this Gemma 3 checkpoint");
            const auto& cfg = VisionConfig::trf;
            VAI_ASSERT(shape.size() == 4);
            VAI_ASSERT(shape[1] == cfg.imageChannels);
            VAI_ASSERT(shape[2] == cfg.imageSize && shape[3] == cfg.imageSize);
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
        const uint32_t tokensPerImage = visionEncoder ? visionEncoder->tokensPerImage : 0;
        {
            const uint32_t* ids =
                modelCfg.hasVision() ? static_cast<const uint32_t*>(promptTokenIds.hostData()) : nullptr;
            uint32_t last = 0;

            for (uint32_t pos = 0; ids && pos < total;)
            {
                if (ids[pos] != ModelConfig::imageTokenId)
                {
                    ++pos;
                    continue;
                }

                if (imageCount <= imageStarts.size() || total - pos < tokensPerImage ||
                    !std::all_of(
                        ids + pos,
                        ids + pos + tokensPerImage,
                        [&](uint32_t id) { return id == ModelConfig::imageTokenId; }))
                    throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

                if (last < pos)
                    blocks.push_back({ pos, true });

                imageStarts.push_back(pos);
                pos += tokensPerImage;
                blocks.push_back({ pos, false });
                last = pos;
            }

            if (imageStarts.size() != imageCount)
                throw std::runtime_error("Gemma 3 image placeholders do not match the image batch");

            if (last < total)
                blocks.push_back({ total, true });
        }

        const uint32_t chunk = prefillChunk;

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
                    chunkInput = vai::Tensor({ n });
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
                chunkImages = vai::Tensor({ chunkImageCount, shape[1], shape[2], shape[3] });
                chunkImages.setData((const float*)images.hostData() + firstImage * imageFloats);
            }

            sync();   // The GPU may still be reading the masks and imageIndex.
            vai::writeCausalMask(causalMask, causalBits, blockEnd);
            vai::writeCausalMask(localMask, causalBits, blockEnd, modelCfg.trf.slidingWindow);

            uint32_t* rows = imageIndex ? (uint32_t*)imageIndex.map() : nullptr;
            for (uint32_t start : chunkStarts)
                for (uint32_t t = 0; t < tokensPerImage; ++t)
                    *rows++ = start + t;

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
    void step(const vai::Tensor& inImages, const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();   // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        decoder.setGateOpen(nextTokenId != nullptr);

        if (nextTokenId)
            topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs;
        if (nextTokenId)
            outputs.emplace("nextTokenId", nextTokenId);

        vai::TensorDict inputs = { { "inTokenIds", inTokenIds } };
        if (inImages.validShape())
            inputs.emplace("inImages", inImages);

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::production::gemma3

#endif   // PRODUCTION_GEMMA3_NET_H
