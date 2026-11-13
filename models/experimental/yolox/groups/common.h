#ifndef EXPERIMENTAL_YOLOX_GROUPS_COMMON_H
#define EXPERIMENTAL_YOLOX_GROUPS_COMMON_H

#include "vai/nodes/convolution.h"
#include <cstdint>


namespace model::experimental::yolox::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Shared by every yolox group: the conv builders the checkpoint is packed for,
// and the two numbers one tier fixes.
/////////////////////////////////////////////////////////////////////////////////////////
inline vai::node::Convolution::Config baseConv(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize,
                                               uint32_t stride, bool acceptResidual = false)
{
    return {
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = kernelSize,
        .stride = stride,
        .acceptResidual = acceptResidual,
        .epilogue = vai::Epilogue::Silu,
    };
}

inline vai::node::Convolution::Config predictionConv(uint32_t inChannels, uint32_t outChannels)
{
    return {
        .inChannels = inChannels,
        .outChannels = outChannels,
        .kernelSize = 1,
        .stride = 1,
    };
}


struct ModelBaseConfig
{
    uint32_t baseChannels;
    uint32_t baseDepth;
};


}   // namespace model::experimental::yolox::group

#endif   // EXPERIMENTAL_YOLOX_GROUPS_COMMON_H
