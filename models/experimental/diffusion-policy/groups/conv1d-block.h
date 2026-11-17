#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H

#include "../nodes/convolution-1d.h"
#include "../nodes/group-norm-mish.h"
#include <cstdint>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Conv1dBlock - lerobot's DiffusionConv1dBlock: Conv1d, GroupNorm, Mish
// Input : in0  [..., Ci, T], film [..., C] / [..., 2C] and residual [..., Co, T] when opened
// Output: out0 [..., Co, T]
//
// out0 = film(mish(groupNorm(conv(in0)))) + residual
//
// conv is Conv1d(Ci, Co, k, padding=k//2) (checkpoint "block.0"), norm its
// GroupNorm(groups, Co) ("block.1": scale / shift) with the Mish fused. The film
// and residual inputs open on the norm for what the residual block applies right
// after the block: FiLM after its first, the residual after its second.
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
        node::Film film = node::Film::None;   // input "film"
        bool acceptResidual = false;          // input "residual"
    };

    Conv1dBlock(Config config)
    : conv({ .inChannels = config.inChannels, .outChannels = config.outChannels, .kernelSize = config.kernelSize })
    , norm({ .groups = config.groups, .film = config.film, .acceptResidual = config.acceptResidual })
    {
        // clang-format off
        conv - norm;
        // clang-format on

        defineSlot("in0", conv.slot("in0"));
        if (config.film != node::Film::None)
            defineSlot("film", norm.slot("film"));
        if (config.acceptResidual)
            defineSlot("residual", norm.slot("residual"));
        defineSlot("out0", norm.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONV1D_BLOCK_H
