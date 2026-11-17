#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_RESIDUAL_BLOCK_1D_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_RESIDUAL_BLOCK_1D_H

#include "vai/nodes/identity.h"
#include "vai/nodes/linear.h"
#include "conv1d-block.h"
#include <cstdint>
#include <optional>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// ConditionalResidualBlock1d - lerobot's DiffusionConditionalResidualBlock1d
// Input : in0  [Ci, T]
//         cond [1, condDim], mish(global_feature): the Mish of cond_encoder, shared by
//              every block, so the UNet runs it once
// Output: out0 [Co, T]
//
// out  = conv1(in0)                                   Conv1dBlock(Ci, Co)
// out  = scale · out + bias, (scale | bias) = condEncoder(cond)   (FiLM, fused in conv1)
// out0 = conv2(out) + residualConv(in0)               Conv1dBlock(Co, Co), residual fused
//
// condEncoder is cond_encoder.1, Linear(condDim -> 2·Co), or -> Co without filmScale
// (lerobot's use_film_scale_modulation: the bias alone). residualConv is the 1x1
// Conv1d(Ci, Co) when Ci != Co; otherwise in0 itself is the residual.
/////////////////////////////////////////////////////////////////////////////////////////
struct ConditionalResidualBlock1d : public vai::NodeGroup
{
    vai::node::Identity input;
    Conv1dBlock conv1;                                  // [Ci, T] -> [Co, T], + FiLM
    vai::node::Linear condEncoder;                      // [1, condDim] -> [1, 2·Co] / [1, Co]
    Conv1dBlock conv2;                                  // [Co, T] -> [Co, T], + residual
    std::optional<node::Convolution1d> residualConv;    // [Ci, T] -> [Co, T]; empty: in0

    struct Config {
        uint32_t inChannels;
        uint32_t outChannels;
        uint32_t condDim;
        uint32_t kernelSize = 3;
        uint32_t groups = 8;
        bool filmScale = false;   // use_film_scale_modulation
    };

    ConditionalResidualBlock1d(Config config)
    : conv1({
          .inChannels = config.inChannels,
          .outChannels = config.outChannels,
          .kernelSize = config.kernelSize,
          .groups = config.groups,
          .film = config.filmScale ? node::Film::ScaleShift : node::Film::Shift,
      })
    , condEncoder({ .inDim = config.condDim, .outDim = config.filmScale ? 2 * config.outChannels : config.outChannels })
    , conv2({
          .inChannels = config.outChannels,
          .outChannels = config.outChannels,
          .kernelSize = config.kernelSize,
          .groups = config.groups,
          .acceptResidual = true,
      })
    {
        if (config.inChannels != config.outChannels)
            residualConv.emplace(node::Convolution1d::Config{
                .inChannels = config.inChannels,
                .outChannels = config.outChannels,
                .kernelSize = 1,
            });

        // clang-format off
        input - conv1 -                    conv2;
        condEncoder -              "film"/conv1;
        input - residualConv - "residual"/conv2;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("cond", condEncoder.slot("in0"));
        defineSlot("out0", conv2.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_RESIDUAL_BLOCK_1D_H
