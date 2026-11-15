#ifndef PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H
#define PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H

// Verbatim copy of setProfileTags() from models/production/llama3/demo.cpp (the
// members it walks are not exposed to Python). Re-copy when the source changes.

#include "models/production/llama3/net.h"
#include <string>

namespace pyvai {
namespace llama3 = ::model::production::llama3;   // the bindings' short name for it
}

namespace pyvai::examples {
namespace {

void setProfileTags(llama3::Net& net)
{
    const auto& trf = net.modelCfg.trf;
    const uint32_t D = trf.embedDim;
    const uint32_t Dkv = trf.headDim * trf.numKVHeads;
    const uint32_t Dff = trf.ffnHiddenDim;
    const uint32_t V = net.modelCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.trf.nBlocks; ++i)
    {
        auto& b = net.decoder.blocks[i];
        b.norm1.setProfileTag("RmsNorm");
        b.mhAttn.projQ.setProfileTag(lin(D, D));
        b.mhAttn.projK.setProfileTag(lin(D, Dkv));
        b.mhAttn.projV.setProfileTag(lin(D, Dkv));
        b.mhAttn.ropeQ->setProfileTag("Rope");
        b.mhAttn.ropeK->setProfileTag("Rope");
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.projOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("RmsNorm");
        b.ffn.gateValue.setProfileTag(lin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("GatedMul");
        b.ffn.out.setProfileTag(lin(Dff, D));
    }
    net.decoder.finalNorm.setProfileTag("RmsNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}

}   // namespace
}   // namespace pyvai::examples

#endif // PYVAI_MODELS_LLAMA3_PROFILE_TAGS_H
