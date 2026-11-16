/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        pi0.5 (3B VLA — PaliGemma 3B and a 300M action expert on adaRMS)
=============================================================================*/

#ifndef PRODUCTION_PI05_NET_H
#define PRODUCTION_PI05_NET_H

#include "groups/action-decoder.h"
#include "models/common.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/patch-embedding.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/cyclic-table-reader.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/linear.h"
#include <cmath>
#include <string>
#include <utility>


namespace model::production::pi05 {


using VisionEncoder = vai::group::Transformer<vai::TransformerType::Classic>;
using LanguageEncoder = vai::group::Transformer<vai::TransformerType::Default>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - pi0.5 (3B VLA) - PaliGemma 3B and a Gemma 300M action expert
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;              // checkpoint folder under data/ckpts/pi05/lerobot/
    uint32_t maxCameras = 3;          // image slots the policy was trained with
    uint32_t maxPromptTokens = 200;   // task and discretized state, as text
    uint32_t vocabSize = 257152;      // 256000 text, 1024 <loc*> and 128 <seg*>
    uint32_t actionDim = 32;          // one action, and the state padded to the same width
    uint32_t actionChunk = 50;        // actions one chunk carries

    VisionTransformerConfig visionTrf = common::so400m14(224);
    vai::TransformerConfig languageTrf = common::gemma("2B");
    vai::TransformerConfig actionTrf = common::gemma("300M");

    // Every camera the policy allows, the prompt and the action chunk: the
    // longest context the cache ever holds. The state rides in the prompt.
    constexpr uint32_t maxContext() const
    {
        return maxCameras * visionTrf.numPatches() + maxPromptTokens + actionChunk;
    }

    // Rows of the action expert's condition: scale, shift and gate for both
    // norms of every block, then scale and shift of its final norm.
    constexpr uint32_t condRows() const { return 6 * actionTrf.nBlocks + 2; }

    static constexpr ModelConfig _3B()
    {
        return {
            .variant = "3B",
        };
    }
};


struct RuntimeConfig
{
    uint32_t denoiseSteps = 10;
    vai::Dtype kvCacheDtype = vai::Dtype::Fp16;
};


struct Net : public vai::NeuralNet
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;

    vai::group::PatchEmbedding patchEmbed;
    VisionEncoder visionEncoder;
    vai::node::Linear projector;
    vai::node::Reshape mergeCameras;

    vai::node::TokenEmbedding promptEmbed;
    vai::node::Concat catCamsPrompt;
    LanguageEncoder languageEncoder;

    vai::node::Linear actionProj;
    vai::node::CyclicTableReader condTable;
    group::ActionDecoder actionDecoder;
    vai::node::Linear velocityHead;

    eva::Buffer kvCacheCounter;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , patchEmbed({
          .imageChannels = modelCfg.visionTrf.imageChannels,
          .imageSize = modelCfg.visionTrf.imageSize,
          .patchSize = modelCfg.visionTrf.patchSize,
          .embedDim = modelCfg.visionTrf.embedDim,
          .allowBatch = true,
      })
    , visionEncoder({ .base = modelCfg.visionTrf, .allowBatch = true })
    , projector({
          .inDim = modelCfg.visionTrf.embedDim,
          .outDim = modelCfg.languageTrf.embedDim,
      })
    , mergeCameras({ -1, modelCfg.languageTrf.embedDim })
    , promptEmbed({
          .vocabSize = modelCfg.vocabSize,
          .embedDim = modelCfg.languageTrf.embedDim,
          .scale = std::sqrt(float(modelCfg.languageTrf.embedDim)),
      })
    , languageEncoder({
          .base = modelCfg.languageTrf,
          .kvCacheSize = modelCfg.maxContext(),
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .gatingLastOut = true,
      })   // Only the prefix KV cache is used.
    , actionProj({
          .inDim = modelCfg.actionDim,
          .outDim = modelCfg.actionTrf.embedDim,
      })
    , condTable({ .numRows = runtimeCfg.denoiseSteps })
    , actionDecoder({
          .base = modelCfg.actionTrf,
          .kvCacheSize = modelCfg.maxContext(),
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
      })
    , velocityHead({
          .inDim = modelCfg.actionTrf.embedDim,
          .outDim = modelCfg.actionDim,
          .acceptResidual = true,
          .scalarScale = -1.0f / runtimeCfg.denoiseSteps,
      })
    {
        // clang-format off
        input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
        input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
        catCamsPrompt - languageEncoder;

        input("inActions") -                         "residual"/velocityHead;
        input("inActions") - actionProj -       actionDecoder - velocityHead - output("outActions");
        input("inActions") - condTable - "cond"/actionDecoder;
        // clang-format on

        languageEncoder.setGateOpen(false);
        kvCacheCounter = vai::createKvCacheCounter(device());
        languageEncoder.setKvCacheCounter(kvCacheCounter);
        actionDecoder.setKvCacheCounter(kvCacheCounter);

        for (uint32_t i = 0; i < modelCfg.languageTrf.nBlocks; ++i)
        {
            auto& encAttn = languageEncoder.blocks[i].mhAttn.attn;
            auto& decAttn = actionDecoder.blocks[i].mhAttn.attn;
            decAttn["K_cache"] = encAttn["K_cache"];
            decAttn["V_cache"] = encAttn["V_cache"];
        }

        velocityHead.setODtype(vai::Dtype::Fp32, dtypeKey());
    }

    uint32_t prefixLength_ = 0;

    // Input: images [cameras, C, S, S], promptTokenIds [T] carrying task and state.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds)
    {
        const uint32_t cameras = images.shape()[0];
        VAI_ASSERT(cameras >= 1 && cameras <= modelCfg.maxCameras);
        VAI_ASSERT(promptTokenIds.shape()[0] <= modelCfg.maxPromptTokens);
        sync();   // The GPU may still be reading kvCacheCounter.

        prefixLength_ = cameras * modelCfg.visionTrf.numPatches() + promptTokenIds.shape()[0];
        *(uint32_t*)kvCacheCounter.map() = 0;
        condTable.rewind();

        (*this)({ { "inImages", images }, { "inPromptTokens", promptTokenIds } }, {}, false);
    }

    // Input: action [actionChunk, actionDim], initialized with noise.
    // Output: denoised actions in the same GPU tensor.
    void decode(vai::Tensor* action)
    {
        VAI_ASSERT(action);

        sync();   // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = prefixLength_;

        for (uint32_t step = 0; step < runtimeCfg.denoiseSteps; ++step)
            (*this)({ { "inActions", *action } }, { { "outActions", action } }, false);
    }
};


}   // namespace model::production::pi05

#endif
