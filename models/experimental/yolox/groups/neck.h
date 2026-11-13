#ifndef EXPERIMENTAL_YOLOX_GROUPS_NECK_H
#define EXPERIMENTAL_YOLOX_GROUPS_NECK_H

#include "csp-layer.h"
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
    vai::node::Convolution lateralConv0;
    vai::node::NearestUpsample upsample0{ 2u };
    vai::node::Concat concatP4;
    CSPLayer c3P4;
    vai::node::Convolution reduceConv1;
    vai::node::NearestUpsample upsample1{ 2u };
    vai::node::Concat concatP3;
    CSPLayer c3P3;
    vai::node::Convolution bottomUpConv2;
    vai::node::Concat concatN3;
    CSPLayer c3N3;
    vai::node::Convolution bottomUpConv1;
    vai::node::Concat concatN4;
    CSPLayer c3N4;

    Neck(const ModelBaseConfig& cfg)
    : lateralConv0(baseConv(16u * cfg.baseChannels, 8u * cfg.baseChannels, 1u, 1u))
    , c3P4(16u * cfg.baseChannels, 8u * cfg.baseChannels, cfg.baseDepth, false)
    , reduceConv1(baseConv(8u * cfg.baseChannels, 4u * cfg.baseChannels, 1u, 1u))
    , c3P3(8u * cfg.baseChannels, 4u * cfg.baseChannels, cfg.baseDepth, false)
    , bottomUpConv2(baseConv(4u * cfg.baseChannels, 4u * cfg.baseChannels, 3u, 2u))
    , c3N3(8u * cfg.baseChannels, 8u * cfg.baseChannels, cfg.baseDepth, false)
    , bottomUpConv1(baseConv(8u * cfg.baseChannels, 8u * cfg.baseChannels, 3u, 2u))
    , c3N4(16u * cfg.baseChannels, 16u * cfg.baseChannels, cfg.baseDepth, false)
    {
        // clang-format off
        lateralConv0 - upsample0 - "in0"/concatP4;
        concatP4 - c3P4 - reduceConv1 - upsample1 - "in0"/concatP3;
        concatP3 - c3P3;

        c3P3       - bottomUpConv2 - "in0"/concatN3;
        reduceConv1                 - "in1"/concatN3;
        concatN3 - c3N3 - bottomUpConv1 - "in0"/concatN4;
        lateralConv0                    - "in1"/concatN4;
        concatN4 - c3N4;
        // clang-format on

        defineSlot("in_p3", concatP3.slot("in1"));
        defineSlot("in_p4", concatP4.slot("in1"));
        defineSlot("in_p5", lateralConv0.slot("in0"));
        defineSlot("out_p3", c3P3.slot("out0"));
        defineSlot("out_p4", c3N3.slot("out0"));
        defineSlot("out_p5", c3N4.slot("out0"));
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_NECK_H
