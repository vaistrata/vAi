#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H

#include "../nodes/convolution-1d.h"
#include "../nodes/group-norm-mish.h"
#include <cstdint>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Conv1dBlock - lerobot's DiffusionConv1dBlock: Conv1d, GroupNorm, Mish
// Input:  in0  [..., Ci, T]
// Output: out0 [..., Co, T]
//
// out0 = mish(groupNorm(conv(in0)))
//
// conv is Conv1d(Ci, Co, k, padding=k//2) (checkpoint "block.0"), norm its
// GroupNorm(groups, Co) ("block.1": scale / shift) with the Mish fused. The FiLM
// and the residual the residual block applies after it are nodes of their own there.
/////////////////////////////////////////////////////////////////////////////////////////
struct Conv1dBlock : public vai::NodeGroup
{
    node::Convolution1d conv;   // [..., Ci, T] -> [..., Co, T]
    node::GroupNormMish norm;   //              -> [..., Co, T]

    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t kernelSize;
        uint32_t groups = 8;
    };

    Conv1dBlock(Config config)
    : conv({ .inChannels = config.inChannels, .outChannels = config.outChannels, .kernelSize = config.kernelSize })
    , norm({ .groups = config.groups })
    {
        // clang-format off
        conv - norm;
        // clang-format on

        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", norm.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H
