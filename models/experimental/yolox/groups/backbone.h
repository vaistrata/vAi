#ifndef EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H
#define EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H

#include "focus-module.h"
#include "csp-layer.h"
#include "spp-bottleneck.h"


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Backbone - CSPDarknet backbone
// Input:   in0     [3, H, W]
// Outputs: out_p3  [4*baseChannels, H/8, W/8]
//          out_p4  [8*baseChannels, H/16, W/16]
//          out_p5  [16*baseChannels, H/32, W/32]
/////////////////////////////////////////////////////////////////////////////////////////
struct Backbone : public vai::NodeGroup
{
    struct DownStage : public vai::NodeGroup
    {
        vai::node::Convolution conv;
        CSPLayer csp;

        DownStage(uint32_t inChannels, uint32_t outChannels, uint32_t repeats, bool shortcut = true)
            : conv(baseConv(inChannels, outChannels, 3u, 2u)), csp(outChannels, outChannels, repeats, shortcut)
        {
            conv - csp;
            defineSlot("in0", conv.slot("in0"));
            defineSlot("out0", csp.slot("out0"));
        }
    };

    struct LastStage : public vai::NodeGroup
    {
        vai::node::Convolution conv;
        SPPBottleneck spp;
        CSPLayer csp;

        LastStage(uint32_t inChannels, uint32_t outChannels, uint32_t repeats)
            : conv(baseConv(inChannels, outChannels, 3u, 2u)), spp(outChannels, outChannels),
              csp(outChannels, outChannels, repeats, false)
        {
            conv - spp - csp;
            defineSlot("in0", conv.slot("in0"));
            defineSlot("out0", csp.slot("out0"));
        }
    };

    FocusModule stem;
    DownStage dark2;
    DownStage dark3;
    DownStage dark4;
    LastStage dark5;

    Backbone(const ModelBaseConfig& cfg)
        : stem(3u, cfg.baseChannels, 3u), dark2(cfg.baseChannels, 2u * cfg.baseChannels, cfg.baseDepth),
          dark3(2u * cfg.baseChannels, 4u * cfg.baseChannels, 3u * cfg.baseDepth),
          dark4(4u * cfg.baseChannels, 8u * cfg.baseChannels, 3u * cfg.baseDepth),
          dark5(8u * cfg.baseChannels, 16u * cfg.baseChannels, cfg.baseDepth)
    {
        stem - dark2 - dark3 - dark4 - dark5;

        defineSlot("in0", stem.slot("in0"));
        defineSlot("out_p3", dark3.slot("out0"));
        defineSlot("out_p4", dark4.slot("out0"));
        defineSlot("out_p5", dark5.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_BACKBONE_H
