/*=============================================================================
  Written by   oneway@vaistrata.com,
               qkrtmddn0507@vaistrata.com
  Model        Gemma 3 (270M / 1B / 4B(VLM) / 12B(VLM))
=============================================================================*/

#ifndef PRODUCTION_GEMMA3_NET_H
#define PRODUCTION_GEMMA3_NET_H

#define TEST_INDEX_COPY

#include "groups/vision-encoder.h"
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
    TransformerConfig decoder;
    uint32_t vocabSize;
    uint32_t maxTrainedContext;
    std::optional<VisionEncoderConfig> vision;

    static constexpr ModelConfig get(std::string_view variant)
    {
        constexpr ModelConfig variants[] = {
            //  variant  {blocks  D  Hq  Hkv  Dff  window  global_RoPE}  vocab  maxContext  vision
            { "270M", { 18, 640, 4, 1, 2048, 512, 1.0f }, 262144, 32768, std::nullopt },
            { "1B", { 26, 1152, 4, 1, 6912, 512, 1.0f }, 262144, 32768, std::nullopt },
            { "4B", { 34, 2560, 8, 4, 10240, 1024, 8.0f }, 262208, 131072, VisionEncoderConfig() },
            { "12B", { 48, 3840, 16, 8, 15360, 1024, 8.0f }, 262208, 131072, VisionEncoderConfig() },
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
    uint32_t prefillChunk = 512;   // 0: the whole prompt in one forward; an image never splits
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;

    std::optional<group::VisionEncoder> visionEncoder;
    std::optional<vai::node::Linear> projector;   // image tokens to the decoder's width
#ifdef TEST_INDEX_COPY
    vai::node::Reshape imageRows;
    vai::node::IndexCopy imageCopy;
#else
    vai::node::RowSliceScatter imageScatter;
#endif

    vai::node::TokenEmbedding embedding;
    group::Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;
    eva::Buffer localMask;   // causalMask spanning the sliding window, the local blocks'
#ifdef TEST_INDEX_COPY
    eva::Buffer imageIndex;
#endif

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.maxTrainedContext))
#ifdef TEST_INDEX_COPY
    , imageRows({ -1, int64_t(modelCfg.decoder.embedDim) })
#else
    , imageScatter({ .sliceSize = modelCfg.vision ? modelCfg.vision->tokensPerImage : 0 })
#endif
    , embedding({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.decoder.embedDim,
          .scale = std::sqrt(float(modelCfg.decoder.embedDim)),
      })
    , decoder({
          .base = modelCfg.decoder,
          .kvCacheSize = maxRuntimeContext,
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .maxForwardTokens = runtimeCfg.prefillChunk ? runtimeCfg.prefillChunk : maxRuntimeContext,
          .gatingLastOut = runtimeCfg.prefillChunk > 0,
      })
    , lmHead({ .inDim = modelCfg.decoder.embedDim, .outDim = modelCfg.vocabSize })
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        if (modelCfg.vision)
        {
            // A chunk holds its images whole.
            VAI_ASSERT(runtimeCfg.prefillChunk == 0 || runtimeCfg.prefillChunk >= modelCfg.vision->tokensPerImage);
            visionEncoder.emplace(*modelCfg.vision);
            projector.emplace(vai::node::Linear::Config{
                .inDim = modelCfg.vision->layerCfg.embedDim,
                .outDim = modelCfg.decoder.embedDim,
            });
        }

        // clang-format off
#ifdef TEST_INDEX_COPY
        input("inTokenIds") - embedding -                           "in0"/imageCopy;
        input("inImages") - visionEncoder - projector - imageRows - "in1"/imageCopy;
        imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");
#else
        input("inTokenIds") - embedding -               "in0"/imageScatter;
        input("inImages") - visionEncoder - projector - "in1"/imageScatter;
        imageScatter - decoder - lmHead - topkSampling - output("nextTokenId");
#endif
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        localMask = vai::createCausalMask(device(), 1, {}, modelCfg.decoder.slidingWindow);
        decoder.setCausalMask(causalMask, localMask);

        if (visionEncoder)
        {
#ifdef TEST_INDEX_COPY
            imageIndex = device().createBuffer({
                .size = MAX_CAUSAL_BLOCKS * modelCfg.vision->tokensPerImage * sizeof(uint32_t),
                .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
                .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
            });
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
            if (!modelCfg.vision || !visionEncoder)
                throw std::runtime_error("image input is not supported by this Gemma 3 checkpoint");
            const auto& cfg = *modelCfg.vision;
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
        const uint32_t tokensPerImage = modelCfg.vision ? modelCfg.vision->tokensPerImage : 0;
        {
            const uint32_t imageTokenId = modelCfg.vision ? modelCfg.vision->imageTokenId : 0;
            const uint32_t* ids = modelCfg.vision ? static_cast<const uint32_t*>(promptTokenIds.hostData()) : nullptr;
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
            vai::writeCausalMask(localMask, causalBits, blockEnd, modelCfg.decoder.slidingWindow);

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

#endif
