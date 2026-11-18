/*=============================================================================
  Written by   oneway@vaistrata.com
  Model        pi0.5 (3B VLA) on two nets: the prefill and the decode each keep
               their own command buffer, so a prefill - decode - prefill loop
               re-records nothing after its first round.
=============================================================================*/

#ifndef EXPERIMENTAL_PI05_NET_H
#define EXPERIMENTAL_PI05_NET_H

#include "models/production/pi05/net.h"
#include <memory>


namespace model::experimental::pi05 {


using production::pi05::LanguageEncoder;
using production::pi05::ModelConfig;
using production::pi05::RuntimeConfig;
using production::pi05::VisionEncoder;
namespace group = production::pi05::group;


/////////////////////////////////////////////////////////////////////////////////////////
// Precision - per tower, the dtypes a block's nodes run at
//   accF32       the projections accumulate in FP32 (NVIDIA's default is the weight's FP16)
//   branchFp16   the branch between the residual stream and its projections back
//                (norms, Q/K/V, attention, FFN hidden) writes FP16; the residual stream
//                itself (the output projections, the gated residuals) stays FP32
//   residualFp16 the residual stream writes FP16 too (the tower's input included)
/////////////////////////////////////////////////////////////////////////////////////////
struct TowerPrecision
{
    bool accF32 = false;
    bool branchFp16 = false;
    bool residualFp16 = false;
};

template <typename Block>
void setBlockPrecision(Block& b, TowerPrecision p, vai::DtypeKey key)
{
    auto& a = b.mhAttn;
    constexpr bool classic = requires { b.ffn.fc1; };   // SigLIP's FFN, else Gemma's gated one
    if (p.accF32)
    {
        for (vai::node::Linear* l : { &a.projQ, &a.projK, &a.projV, &a.projOut })
            l->setAccDtype(vai::Dtype::Fp32, key);
        if constexpr (classic)
        {
            b.ffn.fc1.setAccDtype(vai::Dtype::Fp32, key);
            b.ffn.fc2.setAccDtype(vai::Dtype::Fp32, key);
        }
        else
        {
            b.ffn.gateValue.setAccDtype(vai::Dtype::Fp32, key);
            b.ffn.out.setAccDtype(vai::Dtype::Fp32, key);
        }
    }
    if (p.branchFp16)
    {
        const auto f16 = vai::Dtype::Fp16;
        b.norm1.setODtype(f16, key);
        b.norm2.setODtype(f16, key);
        for (vai::Node* n : { (vai::Node*)&a.projQ, (vai::Node*)&a.projK, (vai::Node*)&a.projV,
                              (vai::Node*)&a.perQ, (vai::Node*)&a.perK, (vai::Node*)&a.perV, (vai::Node*)&a.attn })
            n->setODtype(f16, key);
        if (a.ropeQ)
        {
            a.ropeQ->setODtype(f16, key);
            a.ropeK->setODtype(f16, key);
        }
        if constexpr (classic)
            b.ffn.fc1.setODtype(f16, key);
        else
        {
            b.ffn.gateValue.setODtype(f16, key);
            b.ffn.gatedMul.setODtype(f16, key);
        }
    }
    if (p.residualFp16)
    {
        const auto f16 = vai::Dtype::Fp16;
        a.projOut.setODtype(f16, key);
        if constexpr (classic)
            b.ffn.fc2.setODtype(f16, key);
        else
            b.ffn.out.setODtype(f16, key);
        if constexpr (requires { b.fma1; })   // the action expert's gated residuals
        {
            b.fma1.setODtype(f16, key);
            b.fma2.setODtype(f16, key);
        }
    }
}

inline constexpr TowerPrecision visionPrecision = { .accF32 = true, .branchFp16 = true, .residualFp16 = false };
inline constexpr TowerPrecision languagePrecision = { .accF32 = false, .branchFp16 = true, .residualFp16 = false };
inline constexpr TowerPrecision actionPrecision = { .accF32 = false, .branchFp16 = true, .residualFp16 = true };


/////////////////////////////////////////////////////////////////////////////////////////
// PrefillNet - the cameras and the prompt into the KV cache
/////////////////////////////////////////////////////////////////////////////////////////
struct PrefillNet : public vai::NeuralNet
{
    vai::group::PatchEmbedding patchEmbed;
    VisionEncoder visionEncoder;
    vai::node::Linear projector;
    vai::node::Reshape mergeCameras;

    vai::node::TokenEmbedding promptEmbed;
    vai::node::Concat catCamsPrompt;
    LanguageEncoder languageEncoder;

    eva::Buffer kvCacheCounter;

    PrefillNet(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg)
    : NeuralNet(netCfg)
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
    {
        // clang-format off
        input("inImages") - patchEmbed - visionEncoder - projector - mergeCameras - "in0"/catCamsPrompt;
        input("inPromptTokens")                                     - promptEmbed - "in1"/catCamsPrompt;
        catCamsPrompt - languageEncoder;
        // clang-format on

        languageEncoder.setGateOpen(false);
        kvCacheCounter = vai::createKvCacheCounter(device());
        languageEncoder.setKvCacheCounter(kvCacheCounter);

        for (auto& b : visionEncoder.blocks)
            setBlockPrecision(b, visionPrecision, dtypeKey());
        for (auto& b : languageEncoder.blocks)
            setBlockPrecision(b, languagePrecision, dtypeKey());
        if (languagePrecision.residualFp16)
        {
            projector.setODtype(vai::Dtype::Fp16, dtypeKey());
            promptEmbed.setODtype(vai::Dtype::Fp16, dtypeKey());
            catCamsPrompt.setODtype(vai::Dtype::Fp16, dtypeKey());
        }
        if (visionPrecision.accF32)
        {
            patchEmbed.project.setAccDtype(vai::Dtype::Fp32, dtypeKey());
            projector.setAccDtype(vai::Dtype::Fp32, dtypeKey());
        }
    }

    uint32_t prefixLength_ = 0;

    // Input: images [cameras, C, S, S], promptTokenIds [T] carrying task and state.
    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds, const ModelConfig& modelCfg)
    {
        const uint32_t cameras = images.shape()[0];
        VAI_ASSERT(cameras >= 1 && cameras <= modelCfg.maxCameras);
        VAI_ASSERT(promptTokenIds.shape()[0] <= modelCfg.maxPromptTokens);
        sync();   // The GPU may still be reading kvCacheCounter.

        prefixLength_ = cameras * modelCfg.visionTrf.numPatches() + promptTokenIds.shape()[0];
        *(uint32_t*)kvCacheCounter.map() = 0;

        (*this)({ { "inImages", images }, { "inPromptTokens", promptTokenIds } }, {}, false);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// DecodeNet - the action expert over the prefill's KV cache
/////////////////////////////////////////////////////////////////////////////////////////
struct DecodeNet : public vai::NeuralNet
{
    vai::node::Linear actionProj;
    vai::node::CyclicTableReader condTable;
    group::ActionDecoder actionDecoder;
    vai::node::Linear velocityHead;

    eva::Buffer kvCacheCounter;

    DecodeNet(
        const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg,
        PrefillNet& prefillNet)
    : NeuralNet(netCfg)
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
        input("inActions") -                         "residual"/velocityHead;
        input("inActions") - actionProj -       actionDecoder - velocityHead - output("outActions");
        input("inActions") - condTable - "cond"/actionDecoder;
        // clang-format on

        kvCacheCounter = vai::createKvCacheCounter(device());
        actionDecoder.setKvCacheCounter(kvCacheCounter);

        // The prefill's cache, shared across the two nets: whichever runs first binds it.
        for (uint32_t i = 0; i < modelCfg.languageTrf.nBlocks; ++i)
        {
            auto& encAttn = prefillNet.languageEncoder.blocks[i].mhAttn.attn;
            auto& decAttn = actionDecoder.blocks[i].mhAttn.attn;
            decAttn["K_cache"] = encAttn["K_cache"];
            decAttn["V_cache"] = encAttn["V_cache"];
        }

        velocityHead.setODtype(vai::Dtype::Fp32, dtypeKey());
        for (auto& b : actionDecoder.blocks)
            setBlockPrecision(b, actionPrecision, dtypeKey());
        if (actionPrecision.residualFp16)
            actionProj.setODtype(vai::Dtype::Fp16, dtypeKey());
        if (actionPrecision.accF32)
        {
            actionProj.setAccDtype(vai::Dtype::Fp32, dtypeKey());
            velocityHead.setAccDtype(vai::Dtype::Fp32, dtypeKey());
        }
    }

    // Input: action [actionChunk, actionDim], initialized with noise.
    // Output: denoised actions in the same GPU tensor.
    void decode(vai::Tensor* action, uint32_t prefixLength, uint32_t denoiseSteps)
    {
        VAI_ASSERT(action);

        sync();   // The GPU may still be reading kvCacheCounter.
        *(uint32_t*)kvCacheCounter.map() = prefixLength;
        condTable.rewind();

        for (uint32_t step = 0; step < denoiseSteps; ++step)
            (*this)({ { "inActions", *action } }, { { "outActions", action } }, false);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Net - the two nets behind production's interface
/////////////////////////////////////////////////////////////////////////////////////////
struct Net
{
    ModelConfig modelCfg;
    RuntimeConfig runtimeCfg;

    PrefillNet prefillNet;
    DecodeNet decodeNet;

    // Production's members, so code written against production::pi05::Net reads as it did.
    vai::group::PatchEmbedding& patchEmbed = prefillNet.patchEmbed;
    VisionEncoder& visionEncoder = prefillNet.visionEncoder;
    vai::node::Linear& projector = prefillNet.projector;
    vai::node::TokenEmbedding& promptEmbed = prefillNet.promptEmbed;
    LanguageEncoder& languageEncoder = prefillNet.languageEncoder;
    vai::node::Linear& actionProj = decodeNet.actionProj;
    vai::node::CyclicTableReader& condTable = decodeNet.condTable;
    group::ActionDecoder& actionDecoder = decodeNet.actionDecoder;
    vai::node::Linear& velocityHead = decodeNet.velocityHead;

    Net(const vai::NeuralNetConfig& netCfg, const ModelConfig& modelCfg, const RuntimeConfig& runtimeCfg = {})
    : modelCfg(modelCfg)
    , runtimeCfg(runtimeCfg)
    , prefillNet(sharePipelines(netCfg), modelCfg, runtimeCfg)
    , decodeNet(prefillNet.config(), modelCfg, runtimeCfg, prefillNet)
    {
    }

    const vai::NeuralNetConfig& config() const { return prefillNet.config(); }

    void prefill(const vai::Tensor& images, const vai::Tensor& promptTokenIds)
    {
        const uint32_t recordedPrefix = prefillNet.prefixLength_;
        prefillNet.prefill(images, promptTokenIds, modelCfg);

        // The decode recording tuned its attention for the prefix it was recorded over.
        if (prefillNet.prefixLength_ != recordedPrefix)
            decodeNet.invalidateCommandBuffer();
    }

    void decode(vai::Tensor* action) { decodeNet.decode(action, prefillNet.prefixLength_, runtimeCfg.denoiseSteps); }

    // Input: images and promptTokenIds as prefill() takes them, action as decode() takes it.
    // Output: the action chunk in the same GPU tensor.
    void predict(const vai::Tensor& images, const vai::Tensor& promptTokenIds, vai::Tensor* action)
    {
        prefill(images, promptTokenIds);
        decode(action);
    }

    void sync()
    {
        prefillNet.sync();
        decodeNet.sync();
    }

    void resetProfileRecords()
    {
        prefillNet.resetProfileRecords();
        decodeNet.resetProfileRecords();
    }

private:
    // The device and the pipeline pool, so decodeNet takes the same ones from
    // prefillNet.config(). The buffer pool is not shared: the prefill's scratch,
    // back in the pool once recorded, could become the decode's action tensor,
    // and the next prefill would overwrite it.
    static vai::NeuralNetConfig sharePipelines(vai::NeuralNetConfig netCfg)
    {
        if (!netCfg.device)
            netCfg.device = vai::getVaiDevice();
        if (!netCfg.pipelinePool)
            netCfg.pipelinePool = std::make_shared<vai::PipelinePool>(netCfg.device);
        return netCfg;
    }
};


}   // namespace model::experimental::pi05

#endif
