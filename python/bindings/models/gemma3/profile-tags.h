#ifndef PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H
#define PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from models/production/gemma3/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/production/gemma3/net.h"
#include <cstdint>
#include <string>

namespace pyvai {
namespace gemma3 = ::model::production::gemma3;   // the bindings' short name for it
}

namespace pyvai::examples {
namespace {

void setProfileTags(gemma3::Net& net)
{
    auto towerLin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };
    const auto& trf = net.modelCfg.trf;
    const uint32_t D = trf.embedDim;
    const uint32_t H = trf.numQHeads;
    const uint32_t Hkv = trf.numKVHeads;
    const uint32_t Dh = trf.headDim;
    const uint32_t Dq = H * Dh;
    const uint32_t Dkv = Hkv * Dh;
    const uint32_t Dff = trf.ffnHiddenDim;
    const uint32_t V = net.modelCfg.vocabSize;

    // Vision encoder
    if (net.visionEncoder)
    {
        const auto& v = gemma3::VisionConfig::trf;
        auto& vision = net.visionEncoder.value();

        for (auto& b : vision.encoder.blocks)
        {
            b.norm1.setProfileTag("Vision.LayerNorm");
            b.mhAttn.projQ.setProfileTag(towerLin("Vision", v.embedDim, v.embedDim));
            b.mhAttn.projK.setProfileTag(towerLin("Vision", v.embedDim, v.embedDim));
            b.mhAttn.projV.setProfileTag(towerLin("Vision", v.embedDim, v.embedDim));
            b.mhAttn.attn.setProfileTag("Vision.Attn");
            b.mhAttn.projOut.setProfileTag(towerLin("Vision", v.embedDim, v.embedDim));
            b.norm2.setProfileTag("Vision.LayerNorm");
            b.ffn.fc1.setProfileTag(towerLin("Vision", v.embedDim, v.ffnHiddenDim));
            b.ffn.fc2.setProfileTag(towerLin("Vision", v.ffnHiddenDim, v.embedDim));
        }

        vision.patchEmbed.perPatches.setProfileTag("Vision.Permute");

        vision.patchEmbed.project.setProfileTag(
            towerLin("Vision", v.imageChannels * v.patchSize * v.patchSize, v.embedDim));

        vision.encoder.finalNorm.setProfileTag("Vision.LayerNorm");

        vision.connector.pool.setProfileTag("Vision.Pool");
        vision.connector.tokenNorm.setProfileTag("Vision.RmsNorm");
        vision.connector.projector.setProfileTag(towerLin("Vision", v.embedDim, trf.embedDim));

        net.imageCopy.setProfileTag("Vision.IndexCopy");
    }

    // Text decoder
    const std::string textPrefix = net.visionEncoder ? "Text." : "";

    auto textTag = [&textPrefix](const char* tag) { return textPrefix + tag; };

    auto textLin = [&textPrefix](uint32_t in, uint32_t out) {
        return textPrefix + "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < trf.nBlocks; ++i)
    {
        auto& b = net.decoder.blocks[i];
        b.norm1.setProfileTag(textTag("RmsNorm"));
        b.mhAttn.projQ.setProfileTag(textLin(D, Dq));
        b.mhAttn.projK.setProfileTag(textLin(D, Dkv));
        b.mhAttn.projV.setProfileTag(textLin(D, Dkv));
        b.mhAttn.normQ->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.normK->setProfileTag(textTag("RmsNorm.QK"));
        b.mhAttn.ropeQ->setProfileTag(textTag("Rope"));
        b.mhAttn.ropeK->setProfileTag(textTag("Rope"));

        b.mhAttn.attn.setProfileTag(textTag(gemma3::group::isGlobalBlock(i) ? "Attn.global" : "Attn.local"));

        b.mhAttn.projOut.setProfileTag(textLin(Dq, D));
        b.norm2.setProfileTag(textTag("RmsNorm"));
        b.norm3.setProfileTag(textTag("RmsNorm"));
        b.ffn.gateValue.setProfileTag(textLin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag(textTag("GatedMul"));
        b.ffn.out.setProfileTag(textLin(Dff, D));
        b.norm4.setProfileTag(textTag("RmsNorm"));
    }

    if (net.visionEncoder)
        net.embedding.setProfileTag("Text.Embedding");
    net.decoder.finalNorm.setProfileTag(textTag("RmsNorm"));
    net.lmHead.setProfileTag(textLin(D, V));
    net.topkSampling.setProfileTag(textTag("Sampling"));
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_GEMMA3_PROFILE_TAGS_H
