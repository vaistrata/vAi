#ifndef EXPERIMENTAL_YOLOX_GROUPS_HEAD_H
#define EXPERIMENTAL_YOLOX_GROUPS_HEAD_H

#include "vai/nodes/convolution.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/slice.h"
#include <string>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Head - Three independent decoupled YOLOX prediction branches
// Inputs: p3/p4/p5 feature maps
// Outputs per scale: regobj [B, 5,H,W], cls [B, numClasses,H,W]
/////////////////////////////////////////////////////////////////////////////////////////
struct Head : public vai::NodeGroup
{
    struct Config
    {
        uint32_t baseChannels;
        uint32_t numClasses;
    };

    // Scale indices 0, 1 and 2 correspond to P3, P4 and P5.
    vai::node::Identity input[3];

    vai::node::Convolution stem[3];
    vai::node::Convolution fusedConv[3];
    vai::node::Slice clsSlice[3], regSlice[3];
    vai::node::Convolution clsConv2[3], regConv2[3];
    vai::node::Convolution clsPred[3], regObjPred[3];

    Head(Config config)
    : stem{ { { .inChannels = 4u * config.baseChannels,
                .outChannels = 4u * config.baseChannels,
                .kernelSize = 1u,
                .stride = 1u,
                .actType = vai::ActType::Silu } },
            { { .inChannels = 8u * config.baseChannels,
                .outChannels = 4u * config.baseChannels,
                .kernelSize = 1u,
                .stride = 1u,
                .actType = vai::ActType::Silu } },
            { { .inChannels = 16u * config.baseChannels,
                .outChannels = 4u * config.baseChannels,
                .kernelSize = 1u,
                .stride = 1u,
                .actType = vai::ActType::Silu } } }
    , fusedConv{ { { .inChannels = 4u * config.baseChannels,
                     .outChannels = 8u * config.baseChannels,
                     .kernelSize = 3u,
                     .stride = 1u,
                     .actType = vai::ActType::Silu } },
                 { { .inChannels = 4u * config.baseChannels,
                     .outChannels = 8u * config.baseChannels,
                     .kernelSize = 3u,
                     .stride = 1u,
                     .actType = vai::ActType::Silu } },
                 { { .inChannels = 4u * config.baseChannels,
                     .outChannels = 8u * config.baseChannels,
                     .kernelSize = 3u,
                     .stride = 1u,
                     .actType = vai::ActType::Silu } } }
    , clsSlice{ { { .axis = 1, .start = 0, .count = 4u * config.baseChannels } },
                { { .axis = 1, .start = 0, .count = 4u * config.baseChannels } },
                { { .axis = 1, .start = 0, .count = 4u * config.baseChannels } } }
    , regSlice{ { { .axis = 1, .start = int64_t(4u * config.baseChannels), .count = 4u * config.baseChannels } },
                { { .axis = 1, .start = int64_t(4u * config.baseChannels), .count = 4u * config.baseChannels } },
                { { .axis = 1, .start = int64_t(4u * config.baseChannels), .count = 4u * config.baseChannels } } }
    , clsConv2{ { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } },
                { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } },
                { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } } }
    , regConv2{ { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } },
                { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } },
                { { .inChannels = 4u * config.baseChannels,
                    .outChannels = 4u * config.baseChannels,
                    .kernelSize = 3u,
                    .stride = 1u,
                    .actType = vai::ActType::Silu } } }
    , clsPred{ { { .inChannels = 4u * config.baseChannels,
                   .outChannels = config.numClasses,
                   .kernelSize = 1u,
                   .stride = 1u } },
               { { .inChannels = 4u * config.baseChannels,
                   .outChannels = config.numClasses,
                   .kernelSize = 1u,
                   .stride = 1u } },
               { { .inChannels = 4u * config.baseChannels,
                   .outChannels = config.numClasses,
                   .kernelSize = 1u,
                   .stride = 1u } } }
    , regObjPred{
        { { .inChannels = 4u * config.baseChannels, .outChannels = 5u, .kernelSize = 1u, .stride = 1u } },
        { { .inChannels = 4u * config.baseChannels, .outChannels = 5u, .kernelSize = 1u, .stride = 1u } },
        { { .inChannels = 4u * config.baseChannels, .outChannels = 5u, .kernelSize = 1u, .stride = 1u } }
    }
    {
        const char* scales[3] = { "p3", "p4", "p5" };
        for (uint32_t i = 0; i < 3u; ++i)
        {
            // clang-format off
            input[i] - stem[i] - fusedConv[i];
            fusedConv[i] - clsSlice[i] - clsConv2[i] - clsPred[i];
            fusedConv[i] - regSlice[i] - regConv2[i] - regObjPred[i];
            // clang-format on

            defineSlot(std::string("in_") + scales[i], input[i].slot("in0"));
            defineSlot(std::string("regobj_") + scales[i], regObjPred[i].slot("out0"));
            defineSlot(std::string("cls_") + scales[i], clsPred[i].slot("out0"));
        }
    }
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_HEAD_H
