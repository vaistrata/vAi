#ifndef EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H
#define EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H

#include "../nodes/focus.h"
#include "csp-layer.h"
#include "spp-bottleneck.h"
#include "vai/nodes/convolution.h"
#include <optional>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Backbone - CSPDarknet backbone
// Input:   in0     [B, 3, H, W]
// Outputs: out_p3  [B, 4*baseChannels, H/8, W/8]
//          out_p4  [B, 8*baseChannels, H/16, W/16]
//          out_p5  [B, 16*baseChannels, H/32, W/32]
/////////////////////////////////////////////////////////////////////////////////////////
struct Backbone : public vai::NodeGroup
{
    struct Stage : public vai::NodeGroup
    {
        vai::node::Convolution conv;
        std::optional<SPPBottleneck> spp;
        CSPLayer csp;

        Stage(uint32_t inChannels, uint32_t outChannels, uint32_t repeats, bool shortcut = true, bool hasSPP = false)
        : conv({ .inChannels = inChannels,
                 .outChannels = outChannels,
                 .kernelSize = 3u,
                 .stride = 2u,
                 .epilogue = vai::Epilogue::Silu })
        , csp(outChannels, outChannels, repeats, shortcut)
        {
            if (hasSPP)
                spp.emplace(outChannels, outChannels);

            // clang-format off
            conv - spp - csp;
            // clang-format on
            defineSlot("in0", conv.slot("in0"));
            defineSlot("out0", csp.slot("out0"));
        }
    };

    node::Focus stem;
    Stage stage1, stage2, stage3, stage4;

    Backbone(uint32_t baseChannels, uint32_t baseDepth)
    : stem(3u, baseChannels, 3u)
    , stage1(baseChannels, 2u * baseChannels, baseDepth)
    , stage2(2u * baseChannels, 4u * baseChannels, 3u * baseDepth)
    , stage3(4u * baseChannels, 8u * baseChannels, 3u * baseDepth)
    , stage4(8u * baseChannels, 16u * baseChannels, baseDepth,
             /*shortcut=*/false, /*hasSPP=*/true)
    {
        // clang-format off
        stem - stage1 - stage2 - stage3 - stage4;
        // clang-format on
        defineSlot("in0", stem.slot("in0"));
        defineSlot("out_p3", stage2.slot("out0"));
        defineSlot("out_p4", stage3.slot("out0"));
        defineSlot("out_p5", stage4.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H
