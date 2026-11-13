#ifndef PYVAI_MODELS_PI05_PROFILE_TAGS_H
#define PYVAI_MODELS_PI05_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from models/production/pi05/demo.cpp (the members it
// walks are not exposed to Python). Re-copy when the source changes.

#include "models/production/pi05/net.h"
#include <cstdint>
#include <string>

namespace pyvai {
namespace pi05 = ::model::production::pi05;   // the bindings' short name for it
}

namespace pyvai::examples {
namespace {

/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// a tower collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(pi05::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    // Vision tower — SigLIP, one shape for every layer.
    const auto& modelCfg = net.modelCfg;
    const auto& vitCfg = modelCfg.visionCfg;
    const uint32_t Dv = vitCfg.layerCfg.embedDim;         // 1152
    const uint32_t Dvff = vitCfg.layerCfg.ffnHiddenDim;   // 4304

    for (auto& b : net.visionEncoder.layers)
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
    net.visionEncoder.finalNorm.setProfileTag("Vision.LayerNorm");
    net.projector.setProfileTag(lin("Vision", Dv, modelCfg.languageCfg.layerCfg.embedDim));

    // The two Gemma stacks — same tag scheme, and their widths tell them apart.
    const auto& langCfg = modelCfg.languageCfg.layerCfg;
    const uint32_t Dq = langCfg.numQHeads * langCfg.headDim;     // 2048  (Q projection)
    const uint32_t Dkv = langCfg.numKVHeads * langCfg.headDim;   //  256  (K/V projection)

    auto tagGemma = [&](auto& gemma, const char* tower, uint32_t D, uint32_t Dff) {
        for (auto& b : gemma.layers)
        {
            b.norm1.setProfileTag(std::string(tower) + ".RmsNorm");
            b.mhAttn.projQ.setProfileTag(lin(tower, D, Dq));
            b.mhAttn.projK.setProfileTag(lin(tower, D, Dkv));
            b.mhAttn.projV.setProfileTag(lin(tower, D, Dkv));
            b.mhAttn.ropeQ->setProfileTag(std::string(tower) + ".Rope");
            b.mhAttn.ropeK->setProfileTag(std::string(tower) + ".Rope");
            b.mhAttn.attn.setProfileTag(std::string(tower) + ".Attn");
            b.mhAttn.projOut.setProfileTag(lin(tower, Dq, D));
            b.norm2.setProfileTag(std::string(tower) + ".RmsNorm");
            b.ffn.gateValue.setProfileTag(lin(tower, D, 2 * Dff));
            b.ffn.gatedMul.setProfileTag(std::string(tower) + ".GatedMul");
            b.ffn.out.setProfileTag(lin(tower, Dff, D));
        }
        gemma.finalNorm.setProfileTag(std::string(tower) + ".RmsNorm");
    };

    tagGemma(net.languageEncoder, "Language", langCfg.embedDim, langCfg.ffnHiddenDim);
    tagGemma(
        net.actionDecoder, "Action", modelCfg.actionCfg.layerCfg.embedDim, modelCfg.actionCfg.layerCfg.ffnHiddenDim);

    // The expert's gated residuals, EleAffine in place of Gemma's folded adds.
    for (auto& b : net.actionDecoder.layers)
    {
        b.fma1.setProfileTag("Action.EleAffine");
        b.fma2.setProfileTag("Action.EleAffine");
    }

    net.promptEmbed.setProfileTag("Language.Embedding");
    net.actionProj.setProfileTag("Action.ActionProj");
    net.condTable.setProfileTag("Action.CyclicTableReader");
    net.velocityHead.setProfileTag("Action.VelocityHead");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_PI05_PROFILE_TAGS_H
