/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        pi0 (3B VLA — PaliGemma 3B and a 300M action expert)
=============================================================================*/

#ifndef PRODUCTION_PI0_NET_H
#define PRODUCTION_PI0_NET_H

#include "models/config-base.h"
#include "vai/nodes/token-embedding.h"
#include "vai/groups/patch-embedding.h"
#include "vai/nodes/concat.h"
#include "vai/nodes/slice.h"
#include "vai/nodes/cyclic-table-reader.h"
#include "vai/groups/transformer.h"
#include "vai/nodes/linear.h"
#include <cmath>
#include <string>
#include <utility>


namespace model::production::pi0 {


using VisionTransformer = vai::group::Transformer<vai::TransformerType::Classic>;


using GemmaTransformer = vai::group::Transformer<vai::TransformerType::Default>;


/////////////////////////////////////////////////////////////////////////////////////////
// ModelConfig - pi0 (3B VLA) - PaliGemma 3B and a Gemma 300M action expert
/////////////////////////////////////////////////////////////////////////////////////////
struct ModelConfig
{
    const char* variant;             // checkpoint folder under data/ckpts/pi0/lerobot/
    uint32_t maxCameras = 3;         // image slots the policy was trained with
    uint32_t maxPromptTokens = 48;   // task strings the checkpoint was trained with
    uint32_t actionDim = 32;         // one action, and the state padded to the same width
    uint32_t actionChunk = 50;       // actions one chunk carries
    float timeMinPeriod = 4e-3f;     // fastest sinusoid of the timestep embedding
    float timeMaxPeriod = 4.0f;      // ... and its slowest

    VisionModelConfig visionCfg = siglip::so400m14(224);
    LanguageModelConfig languageCfg = {
        gemma::ModelConfig::_2B().languageCfg,
        257152,   // vocabSize: 256000 text, 1024 <loc*> and 128 <seg*>
    };
    vai::TransformerConfig actionCfg = gemma::ModelConfig::_300M().languageCfg;

    // Every camera the policy allows, the prompt, the state token and the action
    // chunk: the longest context the cache ever holds. A run may carry fewer
    // views, and then the tail of the cache simply stays empty.
    uint32_t maxContext() const { return maxCameras * visionCfg.numPatches() + maxPromptTokens + 1 + actionChunk; }

    static ModelConfig _3B()
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
    VisionTransformer visionEncoder;
    vai::node::Linear projector;
    vai::node::Reshape mergeCameras;

    vai::node::TokenEmbedding promptEmbed;
    vai::node::Concat catCamsPrompt;
    GemmaTransformer languageEncoder;

    vai::node::Linear stateProj;
    vai::node::Linear actionProj;
    vai::node::CyclicTableReader stepTimeBias;
    vai::node::Linear actionTimeIn;
    vai::node::Linear actionTimeOut;
    vai::node::Concat catStateActs;
    GemmaTransformer actionDecoder;
    vai::node::Slice sliceActions;
    vai::node::Linear velocityHead;

    eva::Buffer kvCacheCounter;
    eva::Buffer actionMask;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : NeuralNet(netCfg)
    , modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , patchEmbed({
          .imageChannels = modelCfg.visionCfg.imageChannels,
          .imageSize = modelCfg.visionCfg.imageSize,
          .patchSize = modelCfg.visionCfg.patchSize,
          .embedDim = modelCfg.visionCfg.layerCfg.embedDim,
          .allowBatch = true,
      })
    , visionEncoder(modelCfg.visionCfg.with({ .allowBatch = true, .kvCacheDtype = runtimeCfg.kvCacheDtype }))
    , projector({ .inDim = modelCfg.visionCfg.layerCfg.embedDim, .outDim = modelCfg.languageCfg.layerCfg.embedDim })
    , mergeCameras({ -1, modelCfg.languageCfg.layerCfg.embedDim })
    , promptEmbed({
          .vocabSize = modelCfg.languageCfg.vocabSize,
          .embedDim = modelCfg.languageCfg.layerCfg.embedDim,
          .scale = std::sqrt(float(modelCfg.languageCfg.layerCfg.embedDim)),
      })
    , languageEncoder(modelCfg.languageCfg.with({
          .kvCacheSize = modelCfg.maxContext(),
          .kvCacheDtype = runtimeCfg.kvCacheDtype,
          .gatingLastOut = true,
      }))   // Only the prefix KV cache is used.
    , stateProj({ .inDim = modelCfg.actionDim, .outDim = modelCfg.actionCfg.layerCfg.embedDim })
    , actionProj({ .inDim = modelCfg.actionDim, .outDim = modelCfg.actionCfg.layerCfg.embedDim })
    , stepTimeBias({ .numRows = runtimeCfg.denoiseSteps })
    , actionTimeIn({
          .inDim = modelCfg.actionCfg.layerCfg.embedDim,
          .outDim = modelCfg.actionCfg.layerCfg.embedDim,
          .acceptResidual = true,
          .residualOrder = vai::ResidualOrder::BeforeEpilogue,
          .epilogue = vai::Epilogue::Silu,
      })
    , actionTimeOut({ .inDim = modelCfg.actionCfg.layerCfg.embedDim, .outDim = modelCfg.actionCfg.layerCfg.embedDim })
    , actionDecoder(
          modelCfg.actionCfg.with({ .kvCacheSize = modelCfg.maxContext(), .kvCacheDtype = runtimeCfg.kvCacheDtype }))
    , sliceActions({ .start = 1, .count = modelCfg.actionChunk })
    , velocityHead({
          .inDim = modelCfg.actionCfg.layerCfg.embedDim,
          .outDim = modelCfg.actionDim,
          .acceptResidual = true,
          .scalarScale = -1.0f / runtimeCfg.denoiseSteps,
      })
    {
        // clang-format off
        input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
        input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
        catCamsPrompt - languageEncoder;
        // clang-format on
        languageEncoder.setGateOpen(false);

        // clang-format off
        input("inState") - stateProj - output("outStateToken");

        input("inStateToken")                                                       - "in0"/catStateActs;
        input("inActions") - actionProj              - actionTimeIn - actionTimeOut - "in1"/catStateActs;
        input("inActions") - stepTimeBias - "residual"/actionTimeIn;
        input("inActions")               - "residual"/velocityHead;
        catStateActs - actionDecoder - sliceActions - velocityHead - output("outActions");
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        languageEncoder.setKvCacheCounter(kvCacheCounter);
        actionDecoder.setKvCacheCounter(kvCacheCounter);

        actionMask = vai::createCausalMask(device(), 0, { 1 });
        actionDecoder.setCausalMask(actionMask);

        for (uint32_t i = 0; i < modelCfg.languageCfg.numLayers; ++i)
        {
            auto& encAttn = languageEncoder.layers[i].mhAttn.attn;
            auto& decAttn = actionDecoder.layers[i].mhAttn.attn;
            decAttn["K_cache"] = encAttn["K_cache"];
            decAttn["V_cache"] = encAttn["V_cache"];
        }

        velocityHead.setODtype(vai::Dtype::Fp32, dtypeKey());
    }

    uint32_t prefixLength_ = 0;
    vai::Tensor stateToken_;

    // Input: images [cameras, C, S, S], promptTokenIds [T], state [1, actionDim].
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, const vai::Tensor& state)
    {
        const uint32_t cameras = images.shape()[0];
        VAI_ASSERT(cameras >= 1 && cameras <= modelCfg.maxCameras);
        sync();   // The GPU may still be reading kvCacheCounter.

        prefixLength_ = cameras * modelCfg.visionCfg.numPatches() + promptTokenIds.shape()[0];
        *(uint32_t*)kvCacheCounter.map() = 0;
        stepTimeBias.rewind();

        (*this)({ { "inImages", images }, { "inPromptTokens", promptTokenIds }, { "inState", state } },
                { { "outStateToken", &stateToken_ } },
                false);
    }

    // Input: action [actionChunk, actionDim], initialized with noise.
    // Output: denoised actions in the same GPU tensor.
    void decode(vai::Tensor* action)
    {
        VAI_ASSERT(action);

        sync();   // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = prefixLength_;

        for (uint32_t step = 0; step < runtimeCfg.denoiseSteps; ++step)
            (*this)({ { "inStateToken", stateToken_ }, { "inActions", *action } }, { { "outActions", action } }, false);
    }
};


}   // namespace model::production::pi0

#endif
