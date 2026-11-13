#ifndef PYVAI_MODELS_PALIGEMMA_PROFILE_TAGS_H
#define PYVAI_MODELS_PALIGEMMA_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from models/production/paligemma/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/production/paligemma/net.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace pyvai {
namespace paligemma = ::model::production::paligemma;   // the bindings' short name for it
}

namespace pyvai::examples {
namespace {

// Tags group by kernel shape, so identical-shape ops across a tower collapse
// to one profile row per distinct kernel.
void setProfileTags(paligemma::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    // Vision tower — SigLIP, one shape for every layer.
    const auto& modelCfg = net.modelCfg;
    const auto& vitCfg = modelCfg.visionCfg;
    const uint32_t Dv = vitCfg.layerCfg.embedDim;         // 1152
    const uint32_t Dvff = vitCfg.layerCfg.ffnHiddenDim;   // 4304

    for (auto& b : net.encoder.layers)
    {
        b.norm1.setProfileTag("Vision.LayerNorm");
        b.mhAttn.projQ.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.projK.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.projV.setProfileTag(lin("Vision", Dv, Dv));
        b.mhAttn.attn.setProfileTag("Vision.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Vision", Dv, Dv));
        b.norm2.setProfileTag("Vision.LayerNorm");
        b.ffn.fc1.setProfileTag(lin("Vision", Dv, Dvff));
        b.ffn.fc2.setProfileTag(lin("Vision", Dvff, Dv));
    }
    net.patchEmbed.perPatches.setProfileTag("Vision.Permute");
    net.patchEmbed.project.setProfileTag(lin("Vision", vitCfg.imageChannels * vitCfg.patchSize * vitCfg.patchSize, Dv));
    net.encoder.finalNorm.setProfileTag("Vision.LayerNorm");
    net.projector.setProfileTag(lin("Vision", Dv, modelCfg.languageCfg.layerCfg.embedDim));

    // Text tower — Gemma 2B.
    const auto& langCfg = modelCfg.languageCfg.layerCfg;
    const uint32_t D = langCfg.embedDim;                 // 2048
    const uint32_t Hq = langCfg.numQHeads;               //    8
    const uint32_t Hkv = langCfg.numKVHeads;             //    1 (GQA)
    const uint32_t Dh = langCfg.headDim;                 //  256
    const uint32_t Dq = Hq * Dh;                         // 2048  (Q projection)
    const uint32_t Dkv = Hkv * Dh;                       //  256  (K/V projection)
    const uint32_t Dff = langCfg.ffnHiddenDim;           // 16384
    const uint32_t V = modelCfg.languageCfg.vocabSize;   // 257152

    for (auto& b : net.decoder.layers)
    {
        b.norm1.setProfileTag("Text.RmsNorm");
        b.mhAttn.projQ.setProfileTag(lin("Text", D, Dq));
        b.mhAttn.projK.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.projV.setProfileTag(lin("Text", D, Dkv));
        b.mhAttn.ropeQ->setProfileTag("Text.Rope");
        b.mhAttn.ropeK->setProfileTag("Text.Rope");
        b.mhAttn.attn.setProfileTag("Text.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Text", Dq, D));
        b.norm2.setProfileTag("Text.RmsNorm");
        b.ffn.gateValue.setProfileTag(lin("Text", D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("Text.GatedMul");
        b.ffn.out.setProfileTag(lin("Text", Dff, D));
    }
    net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag("Text.RmsNorm");
    net.lmHead.setProfileTag(lin("Text", D, V));
    net.topkSampling.setProfileTag("Text.Sampling");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_PALIGEMMA_PROFILE_TAGS_H
