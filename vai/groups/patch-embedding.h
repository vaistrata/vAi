#ifndef VAI_GROUPS_PATCH_EMBEDDING_H
#define VAI_GROUPS_PATCH_EMBEDDING_H

#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include "vai/nodes/unflatten.h"
#include <cstdint>
#include <vector>

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// PatchEmbedding - image to one token per non-overlapping patch
// Input: in0 [..., C, H, W], H and W multiples of patchSize
// Output: out0 [..., N, embedDim] = project(patches(in0)), N = Gh * Gw
//
// Gh = H / patchSize and Gw = W / patchSize come from in0, so the image size may
// change from one forward to the next.
//
// The patches come out tileSize x tileSize tiles at a time, raster within a tile and
// tiles in raster order, so a merger downstream finds each tile in consecutive rows.
// tileSize 1 is plain raster order. Gh and Gw must be multiples of tileSize.
//
// A patch row carries (c, kh, kw), the order a [D, C, P, P] patch kernel already has, so
// such a checkpoint loads as a [D, C*P*P] matrix with no repacking.
//
// The axes ahead of C stay their own the whole way, so the images never share a row:
// what one attends to downstream stops at its own patches.
/////////////////////////////////////////////////////////////////////////////////////////
struct PatchEmbedding : public vai::NodeGroup
{
    uint32_t K;   // C * P * P, one patch flattened into a row

    vai::node::Unflatten splitRows;    // [..., C, H, W] -> [..., C, Gh/T, T, P, W]
    vai::node::Unflatten splitCols;    //                -> [..., C, Gh/T, T, P, Gw/T, T, P]
    vai::node::Permute perPatches;     //                -> [..., Gh/T, Gw/T, T, T, C, P, P]
    vai::node::Reshape mergePatches;   //                -> [..., N, K], N = Gh * Gw
    vai::node::Linear project;         // [..., N, K] -> [..., N, D], + bias[N, D] or bias[D]

    struct Config {
        uint32_t imageChannels;
        uint32_t patchSize;
        uint32_t embedDim;
        uint32_t tileSize = 1;   // T: patches leave in T x T tiles; 1 = raster
    };

    PatchEmbedding(Config config)
    : K(config.imageChannels * config.patchSize * config.patchSize)
    , splitRows({ .axis = -2, .sizes = { -1, config.tileSize, config.patchSize } })
    , splitCols({ .axis = -1, .sizes = { -1, config.tileSize, config.patchSize } })
    , perPatches({ 1, 4, 2, 5, 0, 3, 6 })
    , mergePatches({ -1, K }, 7)
    , project({ .inDim = K, .outDim = config.embedDim })
    {
        VAI_ASSERT(config.patchSize >= 1 && config.tileSize >= 1);

        // clang-format off
        splitRows - splitCols - perPatches - mergePatches - project;
        // clang-format on

        defineSlot("in0", splitRows.slot("in0"));
        defineSlot("out0", project.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_PATCH_EMBEDDING_H
