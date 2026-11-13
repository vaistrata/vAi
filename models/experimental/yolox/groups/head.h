#ifndef EXPERIMENTAL_YOLOX_GROUPS_HEAD_H
#define EXPERIMENTAL_YOLOX_GROUPS_HEAD_H

#include "common.h"
#include "vai/nodes/identity.h"
#include <string>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Head - Three independent decoupled YOLOX prediction branches
// Inputs: p3/p4/p5 feature maps
// Outputs per scale: regobj [5,H,W], cls [numClasses,H,W]
/////////////////////////////////////////////////////////////////////////////////////////
struct Head : public vai::NodeGroup
{
    struct Config
    {
        uint32_t baseChannels;
        uint32_t numClasses;
    };

    struct PredictionBranch : public vai::NodeGroup
    {
        vai::node::Convolution stem;
        vai::node::Convolution clsConv1;
        vai::node::Convolution clsConv2;
        vai::node::Convolution clsPred;
        vai::node::Convolution regConv1;
        vai::node::Convolution regConv2;
        vai::node::Convolution regObjPred;

        PredictionBranch(uint32_t inChannels, uint32_t hiddenChannels, uint32_t numClasses)
        : stem(baseConv(inChannels, hiddenChannels, 1u, 1u))
        , clsConv1(baseConv(hiddenChannels, hiddenChannels, 3u, 1u))
        , clsConv2(baseConv(hiddenChannels, hiddenChannels, 3u, 1u))
        , clsPred(predictionConv(hiddenChannels, numClasses))
        , regConv1(baseConv(hiddenChannels, hiddenChannels, 3u, 1u))
        , regConv2(baseConv(hiddenChannels, hiddenChannels, 3u, 1u))
        , regObjPred(predictionConv(hiddenChannels, 5u))
        {
            // clang-format off
            stem - clsConv1 - clsConv2 - clsPred;
            stem - regConv1 - regConv2 - regObjPred;
            // clang-format on

            defineSlot("in0", stem.slot("in0"));
            defineSlot("regobj", regObjPred.slot("out0"));
            defineSlot("cls", clsPred.slot("out0"));
        }
    };

    vai::node::Identity inP3, inP4, inP5;
    PredictionBranch p3;
    PredictionBranch p4;
    PredictionBranch p5;

    Head(Config config)
    : p3(4u * config.baseChannels, 4u * config.baseChannels, config.numClasses)
    , p4(8u * config.baseChannels, 4u * config.baseChannels, config.numClasses)
    , p5(16u * config.baseChannels, 4u * config.baseChannels, config.numClasses)
    {
        // clang-format off
        inP3 - p3;
        inP4 - p4;
        inP5 - p5;
        // clang-format on

        defineSlot("in_p3", inP3.slot("in0"));
        defineSlot("in_p4", inP4.slot("in0"));
        defineSlot("in_p5", inP5.slot("in0"));
        for (const char* output : { "regobj", "cls" })
        {
            defineSlot(std::string(output) + "_p3", p3.slot(output));
            defineSlot(std::string(output) + "_p4", p4.slot(output));
            defineSlot(std::string(output) + "_p5", p5.slot(output));
        }
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_HEAD_H
