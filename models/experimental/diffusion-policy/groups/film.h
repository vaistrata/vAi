#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_FILM_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_FILM_H

#include "vai/nodes/ele-affine.h"
#include "vai/nodes/identity.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/slice.h"
#include "../nodes/group-norm-mish.h"   // node::Film
#include <cstdint>
#include <optional>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// Film - feature-wise linear modulation, per channel over [C, T]
// Input:  in0  [C, T]
//         film [1, 2C] (ScaleShift: scale | shift) or [1, C] (Shift)
// Output: out0 [C, T]
//
// out0 = film[c] · in0 + film[C + c]   (ScaleShift)
// out0 = in0 + film[c]                 (Shift)
//
// lerobot's DiffusionConditionalResidualBlock1d modulation, its cond_encoder output
// split into scale and bias (use_film_scale_modulation) or taken as the bias alone.
// filmView turns the row into a column [2C, 1] / [C, 1] (an alias), the two slices
// take its halves (contiguous blocks, so views when the offset C is aligned), and
// EleAffine broadcasts each [C, 1] over T.
/////////////////////////////////////////////////////////////////////////////////////////
struct Film : public vai::NodeGroup
{
    vai::node::Identity input;
    vai::node::Reshape filmView;             // [1, 2C] -> [2C, 1]  /  [1, C] -> [C, 1]
    std::optional<vai::node::Slice> scale;   // [2C, 1] -> [C, 1]: rows [0, C); ScaleShift only
    std::optional<vai::node::Slice> shift;   // [2C, 1] -> [C, 1]: rows [C, 2C); ScaleShift only
    vai::node::EleAffine affine;             // [C, T] -> [C, T]

    struct Config
    {
        uint32_t channels;   // C
        node::Film mode = node::Film::ScaleShift;
    };

    Film(Config config)
    : filmView({ -1, 1 })
    , affine({ .scaleAsInput = config.mode == node::Film::ScaleShift, .shiftAsInput = true })
    {
        VAI_ASSERT(config.mode != node::Film::None);

        // clang-format off
        input - affine;
        if (config.mode == node::Film::ScaleShift)
        {
            scale.emplace(vai::node::Slice::Config{ .axis = 0, .start = 0, .count = config.channels });
            shift.emplace(vai::node::Slice::Config{ .axis = 0, .start = int64_t(config.channels), .count = config.channels });
            filmView - *scale - "scale"/affine;
            filmView - *shift - "shift"/affine;
        }
        else
            filmView - "shift"/affine;
        // clang-format on

        defineSlot("in0", input.slot("in0"));
        defineSlot("film", filmView.slot("in0"));
        defineSlot("out0", affine.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_FILM_H
