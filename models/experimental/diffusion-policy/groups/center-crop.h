#ifndef EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CENTER_CROP_H
#define EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CENTER_CROP_H

#include "vai/nodes/slice.h"
#include <cstdint>


namespace model::experimental::diffusion_policy::group {


/////////////////////////////////////////////////////////////////////////////////////////
// CenterCrop - torchvision's CenterCrop: the middle cropSize x cropSize of an image
// Input:  in0  [C, S, S],                or [B, C, S, S] with allowBatch
// Output: out0 [C, cropSize, cropSize],  or [B, C, cropSize, cropSize]
//
// The window starts at round((S - cropSize) / 2) on each axis, Python's round (half to
// even) as torchvision's center_crop computes it, e.g. 96 -> 84 starts at 6.
// cropSize may not exceed S: torchvision's padding of a too-small image is not done.
/////////////////////////////////////////////////////////////////////////////////////////
struct CenterCrop : public vai::NodeGroup
{
    vai::node::Slice cropRows;   // [B, C, S, S] -> [B, C, cropSize, S]
    vai::node::Slice cropCols;   //              -> [B, C, cropSize, cropSize]

    struct Config
    {
        uint32_t imageSize;   // S
        uint32_t cropSize;
        bool allowBatch = false;   // true: a leading batch axis, kept the whole way
    };

    // round((S - cropSize) / 2.0), ties to even.
    static constexpr uint32_t cropStart(uint32_t imageSize, uint32_t cropSize)
    {
        const uint32_t diff = imageSize - cropSize;
        const uint32_t half = diff / 2u;
        return (diff % 2u == 1u && half % 2u == 1u) ? half + 1u : half;
    }

    CenterCrop(Config config)
    : cropRows(
          { .axis = config.allowBatch ? 2u : 1u,
            .start = cropStart(config.imageSize, config.cropSize),
            .count = config.cropSize })
    , cropCols(
          { .axis = config.allowBatch ? 3u : 2u,
            .start = cropStart(config.imageSize, config.cropSize),
            .count = config.cropSize })
    {
        VAI_ASSERT(config.cropSize >= 1u && config.cropSize <= config.imageSize);

        // clang-format off
        cropRows - cropCols;
        // clang-format on

        defineSlot("in0", cropRows.slot("in0"));
        defineSlot("out0", cropCols.slot("out0"));
    }
};


}   // namespace model::experimental::diffusion_policy::group

#endif   // EXPERIMENTAL_DIFFUSION_POLICY_GROUPS_CENTER_CROP_H
