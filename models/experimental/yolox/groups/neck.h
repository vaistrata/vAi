#ifndef EXPERIMENTAL_YOLOX_GROUPS_NECK_H
#define EXPERIMENTAL_YOLOX_GROUPS_NECK_H

#include "csp-layer.h"
#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/nearest-upsample.h"
#include "vai/nodes/concat.h"


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Neck - YOLOPAFPN top-down FPN and bottom-up PAN
// Inputs/outputs: p3, p4 and p5 at strides 8, 16 and 32
/////////////////////////////////////////////////////////////////////////////////////////
struct Neck : public vai::NodeGroup
{
    vai::node::Convolution fpnConv0, fpnConv1;   // 1x1 stride-1
    vai::node::Convolution panConv0, panConv1;   // 3x3 stride-2
    CSPLayer fpn0, fpn1;
    CSPLayer pan0, pan1;
    vai::node::NearestUpsample upsample0, upsample1;
    vai::node::Concat fpnConcat0{ { .axis = 1 } }, fpnConcat1{ { .axis = 1 } };
    vai::node::Concat panConcat0{ { .axis = 1 } }, panConcat1{ { .axis = 1 } };

    Neck(uint32_t baseChannels, uint32_t baseDepth)
    : fpnConv0(
          { .inChannels = 16u * baseChannels,
            .outChannels = 8u * baseChannels,
            .kernelSize = 1u,
            .stride = 1u,
            .actType = vai::ActType::Silu })
    , fpnConv1(
          { .inChannels = 8u * baseChannels,
            .outChannels = 4u * baseChannels,
            .kernelSize = 1u,
            .stride = 1u,
            .actType = vai::ActType::Silu })
    , panConv0(
          { .inChannels = 4u * baseChannels,
            .outChannels = 4u * baseChannels,
            .kernelSize = 3u,
            .stride = 2u,
            .actType = vai::ActType::Silu })
    , panConv1(
          { .inChannels = 8u * baseChannels,
            .outChannels = 8u * baseChannels,
            .kernelSize = 3u,
            .stride = 2u,
            .actType = vai::ActType::Silu })
    , fpn0(16u * baseChannels, 8u * baseChannels, baseDepth, false)
    , fpn1(8u * baseChannels, 4u * baseChannels, baseDepth, false)
    , pan0(8u * baseChannels, 8u * baseChannels, baseDepth, false)
    , pan1(16u * baseChannels, 16u * baseChannels, baseDepth, false)
    , upsample0(2u)
    , upsample1(2u)
    {
        // clang-format off
        fpnConv0 - upsample0 - "in0" / fpnConcat0;
        fpnConcat0 - fpn0 - fpnConv1 - upsample1 - "in0" / fpnConcat1;
        fpnConcat1 - fpn1;
        // clang-format on

        // clang-format off
        fpn1 - panConv0 - "in0" / panConcat0;
        fpnConv1 - "in1" / panConcat0;
        panConcat0 - pan0 - panConv1 - "in0" / panConcat1;
        fpnConv0 - "in1" / panConcat1;
        panConcat1 - pan1;
        // clang-format on

        defineSlot("in_p3", fpnConcat1.slot("in1"));
        defineSlot("in_p4", fpnConcat0.slot("in1"));
        defineSlot("in_p5", fpnConv0.slot("in0"));
        defineSlot("out_p3", fpn1.slot("out0"));
        defineSlot("out_p4", pan0.slot("out0"));
        defineSlot("out_p5", pan1.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_NECK_H
