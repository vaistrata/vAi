#ifndef VAI_GROUPS_PATCH_EMBEDDING_H
#define VAI_GROUPS_PATCH_EMBEDDING_H

#include "vai/nodes/linear.h"
#include "vai/nodes/permute.h"
#include "vai/nodes/reshape.h"
#include <cstdint>
#include <vector>

namespace vai::group {


/////////////////////////////////////////////////////////////////////////////////////////
// PatchEmbedding - square image to one token per non-overlapping patch
// Input: in0 [C, S, S], or [B, C, S, S] with allowBatch; S a multiple of patchSize
// Output: out0 [N, embedDim], or [B, N, embedDim] with allowBatch,
//         = project(patches(in0)), N = (S / patchSize)^2
//
// A patch row carries (c, kh, kw), the order a [D, C, P, P] patch kernel already has, so
// such a checkpoint loads as a [D, C*P*P] matrix with no repacking.
//
// The group takes one image. allowBatch adds a leading B that stays an axis of its own
// the whole way, so the images never share a row: what one attends to downstream stops
// at its own patches.
/////////////////////////////////////////////////////////////////////////////////////////
struct PatchEmbedding : public vai::NodeGroup
{
    uint32_t G;   // patches per side
    uint32_t K;   // C * P * P, one patch flattened into a row

    vai::node::Reshape splitPatches;  // [B, C, S, S] -> [B, C, G, P, G, P]
    vai::node::Permute perPatches;    //              -> [B, G, G, C, P, P]
    vai::node::Reshape mergePatches;  //              -> [B, N, K], N = G * G
    vai::node::Linear project;        // [B, N, K] -> [B, N, D], + bias[N, D] or bias[D]

    struct Config {
        uint32_t imageChannels;
        uint32_t imageSize;
        uint32_t patchSize;
        uint32_t embedDim;
        bool allowBatch = false;   // true: a leading batch axis, kept the whole way
    };

    PatchEmbedding(Config config)
    : G(config.imageSize / config.patchSize)
    , K(config.imageChannels * config.patchSize * config.patchSize)
    , splitPatches(config.allowBatch
                       ? std::vector<int64_t>{ -1, config.imageChannels, G, config.patchSize, G, config.patchSize }
                       : std::vector<int64_t>{ config.imageChannels, G, config.patchSize, G, config.patchSize })
    , perPatches(config.allowBatch ? std::vector<uint32_t>{ 0, 2, 4, 1, 3, 5 } : std::vector<uint32_t>{ 1, 3, 0, 2, 4 })
    , mergePatches(config.allowBatch ? std::vector<int64_t>{ 0, G * G, K } : std::vector<int64_t>{ G * G, K })
    , project({ .inDim = K, .outDim = config.embedDim })
    {
        VAI_ASSERT(config.patchSize >= 1);
        VAI_ASSERT(config.imageSize % config.patchSize == 0);

        // clang-format off
        splitPatches - perPatches - mergePatches - project;
        // clang-format on

        defineSlot("in0", splitPatches.slot("in0"));
        defineSlot("out0", project.slot("out0"));
    }
};


}   // namespace vai::group

#endif // VAI_GROUPS_PATCH_EMBEDDING_H
