#ifndef VAI_GROUPS_PATCH_EMBEDDING_H
#define VAI_GROUPS_PATCH_EMBEDDING_H

#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/slice.h"
#include <cstdint>
#include <vector>

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// PatchEmbedding - square image to one token per non-overlapping patch
// Input: in0 [..., C, S, S]
// Output: out0 [..., N, embedDim] = project(patches(in0)), N = (S / patchSize)^2
//
// The pixels past the last whole patch drop, as a stride-patchSize conv drops
// them; when S is a multiple of patchSize the crop passes in0 through.
//
// A patch row carries (c, kh, kw), the order a [D, C, P, P] patch kernel already has, so
// such a checkpoint loads as a [D, C*P*P] matrix with no repacking.
//
// The axes ahead of C stay their own the whole way, so the images never share a row:
// what one attends to downstream stops at its own patches.
/////////////////////////////////////////////////////////////////////////////////////////
struct PatchEmbedding : public vai::NodeGroup
{
    uint32_t G;   // patches per side
    uint32_t K;   // C * P * P, one patch flattened into a row

    vai::node::Slice cropRows;         // [..., C, S, S] -> [..., C, G * P, S]
    vai::node::Slice cropCols;         //                -> [..., C, G * P, G * P]
    vai::node::Reshape splitPatches;   //                -> [..., C, G, P, G, P]
    vai::node::Permute perPatches;     //                -> [..., G, G, C, P, P]
    vai::node::Reshape mergePatches;   //                -> [..., N, K], N = G * G
    vai::node::Linear project;         // [..., N, K] -> [..., N, D], + bias[N, D] or bias[D]

    struct Config {
        uint32_t imageChannels;
        uint32_t imageSize;
        uint32_t patchSize;
        uint32_t embedDim;
    };

    PatchEmbedding(Config config)
    : G(config.imageSize / config.patchSize)
    , K(config.imageChannels * config.patchSize * config.patchSize)
    , cropRows({ .axis = -2, .start = 0, .count = G * config.patchSize })
    , cropCols({ .axis = -1, .start = 0, .count = G * config.patchSize })
    , splitPatches({ config.imageChannels, G, config.patchSize, G, config.patchSize }, 3)
    , perPatches({ 1, 3, 0, 2, 4 })
    , mergePatches({ G * G, K }, 5)
    , project({ .inDim = K, .outDim = config.embedDim })
    {
        VAI_ASSERT(config.patchSize >= 1 && G >= 1);

        // clang-format off
        cropRows - cropCols - splitPatches - perPatches - mergePatches - project;
        // clang-format on

        defineSlot("in0", cropRows.slot("in0"));
        defineSlot("out0", project.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_PATCH_EMBEDDING_H
