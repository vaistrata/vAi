#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_SPATIAL_SOFTMAX_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_SPATIAL_SOFTMAX_H

#include "vai/nodes/linear.h"
#include "vai/nodes/reshape.h"
#include "../nodes/softmax.h"
#include <cstdint>
#include <vector>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// SpatialSoftmax - spatial soft-argmax (Finn et al., "Deep Spatial Autoencoders")
// Input : in0  [K, H, W], or [B, K, H, W] with allowBatch
// Output: out0 [K, 2],    or [B, K, 2]    with allowBatch
//
// out0 = softmax(in0 as [K, H*W], dim=-1) @ pos_grid
//
// Each channel's softmax over H*W weights the normalized (x, y) grid in [-1, 1], so a
// channel yields its activation's center of mass, one keypoint per channel.
//
// pos_grid [H*W, 2] is np.meshgrid(linspace(-1, 1, W), linspace(-1, 1, H)): x runs
// along W, y along H. It is a constant, so the group fills expectedXy's weight
// (pos_grid^T, [2, H*W]) itself; the loader leaves it alone.
//
// robomimic / lerobot's num_kp projection (the 1x1 conv "nets") is not part of it:
// the caller wires that conv ahead of in0, and the keypoint count is its outChannels.
/////////////////////////////////////////////////////////////////////////////////////////
struct SpatialSoftmax : public vai::NodeGroup
{
    uint32_t HW;   // H * W

    vai::node::Reshape flatten;     // [B, K, H, W] -> [B, K, H*W]
    node::Softmax softmax;          //              -> [B, K, H*W], over H*W
    vai::node::Linear expectedXy;   //              -> [B, K, 2],   @ pos_grid

    struct Config {
        uint32_t channels;   // K: one keypoint per channel
        uint32_t height;
        uint32_t width;
        bool allowBatch = false;   // true: a leading batch axis, kept the whole way
    };

    SpatialSoftmax(Config config)
    : HW(config.height * config.width)
    , flatten(config.allowBatch ? std::vector<int64_t>{ 0, config.channels, HW }
                                : std::vector<int64_t>{ config.channels, HW })
    , softmax({ .dim = -1 })
    , expectedXy({ .inDim = HW, .outDim = 2 })
    {
        VAI_ASSERT(HW >= 1);

        // clang-format off
        flatten - softmax - expectedXy;
        // clang-format on

        expectedXy["weight"] = vai::Tensor({ 2u, HW }, vai::Dtype::Fp32)
                                   .setFloats(posGridT(config.height, config.width));

        defineSlot("in0", flatten.slot("in0"));
        defineSlot("out0", expectedXy.slot("out0"));
    }

    // np.linspace(-1, 1, n) in double, then .float(), as the torch module builds it.
    static std::vector<double> linspace(uint32_t n)
    {
        std::vector<double> v(n, -1.0);
        if (n > 1)
        {
            const double step = 2.0 / double(n - 1);
            for (uint32_t i = 0; i < n; ++i)
                v[i] = -1.0 + double(i) * step;
            v[n - 1] = 1.0;
        }
        return v;
    }

    // pos_grid^T [2, H*W]: row 0 the x of each (h, w), row 1 its y.
    static std::vector<float> posGridT(uint32_t H, uint32_t W)
    {
        const std::vector<double> xs = linspace(W), ys = linspace(H);
        std::vector<float> grid(2u * H * W);
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t w = 0; w < W; ++w)
            {
                grid[h * W + w] = float(xs[w]);
                grid[H * W + h * W + w] = float(ys[h]);
            }
        return grid;
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_SPATIAL_SOFTMAX_H
