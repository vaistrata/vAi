#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_UNET_1D_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_UNET_1D_H

#include "vai/nodes/concat.h"
#include "vai/nodes/permute.h"
#include "../nodes/conv-transpose-1d.h"
#include "conditional-residual-block-1d.h"
#include "conv1d-block.h"
#include "diffusion-step-encoder.h"
#include <cstdint>
#include <optional>
#include <vector>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// ConditionalUnet1d - lerobot's DiffusionConditionalUnet1d: one denoising step
// Input:  in0        [T, A]   the noisy action sequence, T = horizon, A = action dim
//         timestep   [1]      the denoising timestep
//         globalCond [1, G]   the observation encoder's global_cond
// Output: out0       [T, A]   the model's prediction (noise, or the sample, per prediction_type)
//
// cond = mish(cat(stepEncoder(timestep), globalCond))         [1, D + G]
// x    = in0 as [A, T], then with L = downDims.size() levels:
//   down l    : x = res(res(x, cond), cond) -> skip[l];  x = downsample(x)   (l < L-1: k3 s2)
//   mid       : x = res(res(x, cond), cond)
//   up u < L-1: x = cat(x, skip[L-1-u]) on channels;  x = upsample(res(res(x, cond), cond))
//   final     : x = finalConv(finalBlock(x)), back to [T, A]
//
// As lerobot builds it: every up level upsamples (its is_last never holds), and
// skip[0] goes unused, so T halves L-1 times and doubles L-1 times (pushT 16 -> 4
// -> 16). The Mish each block's cond_encoder starts with sees the same cond, so
// condMish runs it once for all of them.
/////////////////////////////////////////////////////////////////////////////////////////
struct ConditionalUnet1d : public vai::NodeGroup
{
    vai::node::Permute toChannels;      // [T, A] -> [A, T]
    DiffusionStepEncoder stepEncoder;   // [1] -> [1, D]
    vai::node::Concat catCond;          // [1, D] + [1, G] -> [1, D + G]   (global_feature)
    node::Mish condMish;                //                -> [1, D + G]

    std::vector<ConditionalResidualBlock1d> downBlocks;            // two per level: down_modules.l.{0, 1}
    std::vector<std::optional<node::Convolution1d>> downsamples;   // down_modules.l.2; the last level's empty
    std::vector<ConditionalResidualBlock1d> midBlocks;             // mid_modules.{0, 1}
    std::vector<vai::node::Concat> upCats;                         // cat(x, skip) on channels
    std::vector<ConditionalResidualBlock1d> upBlocks;              // two per level: up_modules.u.{0, 1}
    std::vector<node::ConvTranspose1d> upsamples;                  // up_modules.u.2

    Conv1dBlock finalBlock;          // final_conv.0: [C0, T] -> [C0, T]
    node::Convolution1d finalConv;   // final_conv.1: [C0, T] -> [A, T]
    vai::node::Permute toTime;       // [A, T] -> [T, A]

    struct Config
    {
        uint32_t actionDim;               // A
        uint32_t globalCondDim;           // G
        std::vector<uint32_t> downDims;   // channels per level, e.g. { 512, 1024, 2048 }
        uint32_t kernelSize = 5;
        uint32_t groups = 8;           // n_groups
        uint32_t stepEmbedDim = 128;   // D, diffusion_step_embed_dim
        bool filmScale = true;         // use_film_scale_modulation
    };

    ConditionalUnet1d(const Config& config)
    : toChannels({ 1, 0 })
    , stepEncoder({ .embedDim = config.stepEmbedDim })
    , catCond({ .axis = -1, .inputCount = 2 })
    , condMish()
    , finalBlock({
          .inChannels = config.downDims.front(),
          .outChannels = config.downDims.front(),
          .kernelSize = config.kernelSize,
          .groups = config.groups,
      })
    , finalConv({ .inChannels = config.downDims.front(), .outChannels = config.actionDim, .kernelSize = 1 })
    , toTime({ 1, 0 })
    {
        const std::vector<uint32_t>& dims = config.downDims;
        const uint32_t L = uint32_t(dims.size());
        VAI_ASSERT(L >= 1u);
        const uint32_t condDim = config.stepEmbedDim + config.globalCondDim;
        auto block = [&](uint32_t in, uint32_t out) {
            return ConditionalResidualBlock1d::Config{
                .inChannels = in,
                .outChannels = out,
                .condDim = condDim,
                .kernelSize = config.kernelSize,
                .groups = config.groups,
                .filmScale = config.filmScale,
            };
        };

        // Built in place: a block's slots point back at it, so it cannot move.
        downBlocks.reserve(2 * size_t(L));
        downsamples.reserve(L);
        for (uint32_t l = 0; l < L; ++l)
        {
            const uint32_t in = l == 0 ? config.actionDim : dims[l - 1];
            downBlocks.emplace_back(block(in, dims[l]));
            downBlocks.emplace_back(block(dims[l], dims[l]));
            if (l + 1 < L)
                downsamples.emplace_back(
                    std::in_place,
                    node::Convolution1d::Config{
                        .inChannels = dims[l], .outChannels = dims[l], .kernelSize = 3, .stride = 2 });
            else
                downsamples.emplace_back(std::nullopt);
        }

        midBlocks.reserve(2);
        midBlocks.emplace_back(block(dims.back(), dims.back()));
        midBlocks.emplace_back(block(dims.back(), dims.back()));

        upCats.reserve(L - 1);
        upBlocks.reserve(2 * size_t(L - 1));
        upsamples.reserve(L - 1);
        for (uint32_t u = 0; u + 1 < L; ++u)
        {
            const uint32_t in = dims[L - 1 - u];   // the level below, and its skip
            const uint32_t out = dims[L - 2 - u];
            upCats.emplace_back(vai::node::Concat::Config{ .axis = -2, .inputCount = 2 });
            upBlocks.emplace_back(block(2 * in, out));
            upBlocks.emplace_back(block(out, out));
            upsamples.emplace_back(node::ConvTranspose1d::Config{
                .inChannels = out, .outChannels = out, .kernelSize = 4, .stride = 2, .padding = 1 });
        }

        // clang-format off
        // Condition: shared by every residual block.
        stepEncoder - "in0"/catCond - condMish;
        for (auto* blocks : { &downBlocks, &midBlocks, &upBlocks })
            for (ConditionalResidualBlock1d& b : *blocks)
                condMish - "cond"/b;

        // Down: each level hands its output to the next level, or to the mid blocks.
        toChannels - downBlocks[0];
        for (uint32_t l = 0; l < L; ++l)
        {
            const size_t b = 2 * size_t(l);
            vai::NodeGroup& next = l + 1 < L ? static_cast<vai::NodeGroup&>(downBlocks[b + 2]) : midBlocks[0];
            downBlocks[b] - downBlocks[b + 1] - downsamples[l] - next;
        }
        midBlocks[0] - midBlocks[1];

        // Up: cat(x, skip) -> two blocks -> upsample, into the next cat or the final block.
        if (L == 1)
            midBlocks[1] - finalBlock;
        else
            midBlocks[1] - "in0"/upCats[0];
        for (uint32_t u = 0; u + 1 < L; ++u)
        {
            downBlocks[2 * size_t(L - 1 - u) + 1] - "in1"/upCats[u];
            upCats[u] - upBlocks[2 * size_t(u)] - upBlocks[2 * size_t(u) + 1] - upsamples[u];
            if (u + 2 < L)
                upsamples[u] - "in0"/upCats[u + 1];
            else
                upsamples[u] - finalBlock;
        }
        finalBlock - finalConv - toTime;
        // clang-format on

        defineSlot("in0", toChannels.slot("in0"));
        defineSlot("timestep", stepEncoder.slot("in0"));
        defineSlot("globalCond", catCond.slot("in1"));
        defineSlot("out0", toTime.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CONDITIONAL_UNET_1D_H
