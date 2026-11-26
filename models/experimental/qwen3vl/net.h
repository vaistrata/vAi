/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        Qwen3-VL (2B / 4B / 8B, Instruct)
=============================================================================*/

#ifndef EXPERIMENTAL_QWEN3VL_NET_H
#define EXPERIMENTAL_QWEN3VL_NET_H

#include "groups/vision-encoder.h"
#include "groups/transformer.h"
#include "vai/nodes/index-copy.h"
#include "vai/nodes/linear.h"
#include "vai/nodes/token-embedding.h"
#include "vai/nodes/topk-sampling.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


namespace model::experimental::qwen3vl {


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - Qwen3-VL (2B / 4B / 8B) - a ViT with DeepStack taps and the Qwen3 decoder
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;   // checkpoint folder under data/ckpts/qwen3vl/qwen/
    VisionConfig vision;
    vai::TransformerConfig trf;
    bool tiedHead;   // lmHead shares the token embedding's weights

    static constexpr uint32_t vocabSize = 151936;
    static constexpr uint32_t maxTrainedContext = 262144;
    static constexpr uint32_t imageTokenId = 151655;   // <|image_pad|>, the placeholder an image's tokens replace

    static constexpr ModelConfig get(std::string_view variant)
    {
        constexpr ModelConfig variants[] = {
            //  variant  vision vm(blocks  D  Dff  deepstack)  text lm(blocks  D  Hq  Dff)  tiedHead
            { "2B", vm(24, 1024, 4096, { 5, 11, 17 }), lm(28, 2048, 16, 6144), true },
            { "4B", vm(24, 1024, 4096, { 5, 11, 17 }), lm(36, 2560, 32, 9728), true },
            { "8B", vm(27, 1152, 4304, { 8, 16, 24 }), lm(36, 4096, 32, 12288), false },
        };
        for (const ModelConfig& v : variants)
            if (variant == v.variant)
                return v;
        throw std::invalid_argument("unknown Qwen3-VL variant");
    }
};


struct RuntimeConfig
{
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
    uint32_t maxRuntimeContext = uint32_t(-1);
    uint32_t imageSize = 768;   // every image is squashed to this square, a multiple of 32
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;
    uint32_t maxRuntimeContext;

    group::VisionEncoder visionEncoder;
    vai::node::IndexCopy imageCopy;

    vai::node::TokenEmbedding embedding;
    group::Transformer decoder;
    vai::node::Linear lmHead;
    vai::node::TopKSampling topkSampling;

    eva::Buffer kvCacheCounter;
    eva::Buffer causalMask;
    eva::Buffer imageIndex;       // the image tokens' rows in this forward
    eva::Buffer tokenPositions;   // [T, 3], each token's MRoPE (t, h, w)
    eva::Buffer gridWidth;        // [1], patches per row of an image

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , maxRuntimeContext(std::min(runtimeCfg.maxRuntimeContext, modelCfg.maxTrainedContext))
    , visionEncoder({
          .base = modelCfg.vision,
          .imageSize = runtimeCfg.imageSize,
          .outDim = modelCfg.trf.embedDim,
      })
    , embedding({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.trf.embedDim,
      })
    , decoder({
          .base = modelCfg.trf,
          .kvCacheSize = maxRuntimeContext,
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .deepstackCount = uint32_t(modelCfg.vision.deepstack.size()),
      })
    , lmHead({ .inDim = modelCfg.trf.embedDim, .outDim = modelCfg.vocabSize })
    {
        this->runtimeCfg.maxRuntimeContext = maxRuntimeContext;

        // clang-format off
        input("inTokenIds") - embedding -   "in0"/imageCopy;
        input("inImages") - visionEncoder - "in1"/imageCopy;
        imageCopy - decoder - lmHead - topkSampling - output("nextTokenId");

        for (size_t j = 0; j < modelCfg.vision.deepstack.size(); ++j)
        {
            const std::string tap = "deep" + std::to_string(j);
            visionEncoder/tap - tap/decoder;
        }
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        decoder.setKvCacheCounter(kvCacheCounter);

        causalMask = vai::createCausalMask(device(), 1);
        decoder.setCausalMask(causalMask);

        imageIndex = hostBuffer(maxRuntimeContext);
        imageCopy.setIndex(imageIndex);
        decoder.setImageIndex(imageIndex);

        tokenPositions = hostBuffer(maxRuntimeContext * 3);
        decoder.setPositions(tokenPositions);

        gridWidth = vai::createKvCacheCounter(device());   // a host-visible uniform uint
        *(uint32_t*)gridWidth.map() = visionEncoder.patchesPerSide;
        visionEncoder.setGridWidth(gridWidth);
    }

    uint32_t tokensSoFar_ = 0;
    uint32_t nextPosition_ = 0;   // the MRoPE position the next text token takes

    // Input: images [B, C, S, S]; empty for text only. promptTokenIds [T], on the host.
    // Output: nextTokenId [1], the first generated token.
    // Images must match placeholder blocks in prompt order.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, vai::Tensor* nextTokenId)
    {
        VAI_ASSERT(nextTokenId);
        VAI_ASSERT(promptTokenIds.hasHostData());

        const uint32_t total = promptTokenIds.shape()[0];
        const auto& shape = images.shape();
        const uint32_t imageCount = images.validShape() ? shape[0] : 0;
        VAI_ASSERT(tokensSoFar_ + total <= maxRuntimeContext);

        if (imageCount > 0)
        {
            VAI_ASSERT(shape.size() == 4);
            VAI_ASSERT(shape[1] == modelCfg.vision.trf.imageChannels);
            VAI_ASSERT(shape[2] == runtimeCfg.imageSize && shape[3] == runtimeCfg.imageSize);
        }

        sync();   // The GPU may still be reading the positions and imageIndex.

        // A text token takes the next position on all three axes. An image's tokens
        // share t, spread over h and w by their row and column of its th x tw token
        // grid, and move the text after them on by the longer side.
        const uint32_t* ids = static_cast<const uint32_t*>(promptTokenIds.hostData());
        uint32_t* pos = (uint32_t*)tokenPositions.map();
        uint32_t* rows = (uint32_t*)imageIndex.map();
        uint32_t p = nextPosition_;
        uint32_t image = 0;
        const uint32_t pixelsPerToken = modelCfg.vision.trf.patchSize * VisionConfig::mergeSize;
        const uint32_t th = imageCount > 0 ? shape[2] / pixelsPerToken : 0;
        const uint32_t tw = imageCount > 0 ? shape[3] / pixelsPerToken : 0;
        const uint32_t tokensPerImage = th * tw;

        for (uint32_t n = 0; n < total;)
        {
            if (ids[n] != ModelConfig::imageTokenId)
            {
                pos[3 * n] = pos[3 * n + 1] = pos[3 * n + 2] = p++;
                ++n;
                continue;
            }

            if (imageCount == image || total - n < tokensPerImage ||
                !std::all_of(
                    ids + n, ids + n + tokensPerImage, [](uint32_t id) { return id == ModelConfig::imageTokenId; }))
                throw std::runtime_error("Qwen3-VL image placeholders do not match the image batch");

            for (uint32_t r = 0; r < th; ++r)
                for (uint32_t c = 0; c < tw; ++c)
                {
                    const uint32_t k = n + r * tw + c;
                    pos[3 * k] = p;
                    pos[3 * k + 1] = p + r;
                    pos[3 * k + 2] = p + c;
                    *rows++ = k;
                }

            p += std::max(th, tw);
            n += tokensPerImage;
            ++image;
        }

        if (image != imageCount)
            throw std::runtime_error("Qwen3-VL image placeholders do not match the image batch");

        nextPosition_ = p;
        step(images, promptTokenIds, nextTokenId);
    }

    // Input/output: inoutTokenId [1], previous token replaced by the next token.
    void decode(vai::Tensor* inoutTokenId)
    {
        VAI_ASSERT(inoutTokenId);

        sync();   // The GPU may still be reading the positions.
        uint32_t* pos = (uint32_t*)tokenPositions.map();
        pos[0] = pos[1] = pos[2] = nextPosition_++;

        step({}, *inoutTokenId, inoutTokenId);
    }

    void resetCache()
    {
        sync();   // The GPU may still be reading kvCacheCounter.
        tokensSoFar_ = 0;
        nextPosition_ = 0;
        *(uint32_t*)kvCacheCounter.map() = 0;
    }

private:
    eva::Buffer hostBuffer(uint32_t words)
    {
        return device().createBuffer({
            .size = words * sizeof(uint32_t),
            .usage = eva::BUFFER_USAGE::STORAGE_BUFFER,
            .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE | eva::MEMORY_PROPERTY::HOST_COHERENT,
        });
    }

    void step(const vai::Tensor& inImages, const vai::Tensor& inTokenIds, vai::Tensor* nextTokenId)
    {
        sync();   // The GPU may still be reading kvCacheCounter.

        *(uint32_t*)kvCacheCounter.map() = tokensSoFar_;
        tokensSoFar_ += inTokenIds.shape()[0];

        topkSampling.setUniform((float)std::rand() / ((float)RAND_MAX + 1.0f));

        vai::TensorRefDict outputs = { { "nextTokenId", nextTokenId } };

        vai::TensorDict inputs = { { "inTokenIds", inTokenIds } };
        if (inImages.validShape())
            inputs.emplace("inImages", inImages);

        (*this)(std::move(inputs), std::move(outputs), false);
    }
};


}   // namespace model::experimental::qwen3vl

#endif   // EXPERIMENTAL_QWEN3VL_NET_H
