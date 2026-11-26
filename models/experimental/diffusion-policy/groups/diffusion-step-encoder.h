#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_DIFFUSION_STEP_ENCODER_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_DIFFUSION_STEP_ENCODER_H

#include "vai/nodes/linear.h"
#include "../nodes/mish.h"
#include "../nodes/sinusoidal-pos-emb.h"
#include <cstdint>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// DiffusionStepEncoder - lerobot's diffusion_step_encoder
// Input:  in0  [1] the denoising timestep
// Output: out0 [1, D], D = embedDim
//
// out0 = fc2(mish(fc1(posEmb(in0))))
//
// posEmb is DiffusionSinusoidalPosEmb(D) ("0"), fc1 Linear(D -> 4D) ("1"),
// fc2 Linear(4D -> D) ("3").
/////////////////////////////////////////////////////////////////////////////////////////
struct DiffusionStepEncoder : public vai::NodeGroup
{
    node::SinusoidalPosEmb posEmb;   // [1] -> [1, D]
    vai::node::Linear fc1;           //     -> [1, 4D]
    node::Mish mish;                 //     -> [1, 4D]
    vai::node::Linear fc2;           //     -> [1, D]

    struct Config {
        uint32_t embedDim;   // diffusion_step_embed_dim
    };

    DiffusionStepEncoder(Config config)
    : posEmb({ .dim = config.embedDim })
    , fc1({ .inDim = config.embedDim, .outDim = 4 * config.embedDim })
    , mish({ .inPlace = true })
    , fc2({ .inDim = 4 * config.embedDim, .outDim = config.embedDim })
    {
        // clang-format off
        posEmb - fc1 - mish - fc2;
        // clang-format on

        defineSlot("in0", posEmb.slot("in0"));
        defineSlot("out0", fc2.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_DIFFUSION_STEP_ENCODER_H
