#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/deformable-attention.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pyvai {

static void bindDeformableAttention(nb::module_& m)
{
    using C = model::experimental::rtdetrv2::node::DeformableAttention::Config;
    bindNode<model::experimental::rtdetrv2::node::DeformableAttention>(
        m,
        "DeformableAttentionNode",
        "RT-DETRv2 multi-scale deformable attention core. Inputs values [B, S, M, D] fp32 or "
        "fp16, boxes [B, Q, R, C], offsets [B, Q, M, P, 2], logits [B, Q, M, P], all fp32; "
        "out0 [B, Q, M*D] fp32. S = sum of H*W and P = sum of point_counts over the levels, "
        "in order. R is 1 or the level count; C = 4 boxes (cx, cy, w, h) sample at "
        "xy + offsets / P_l * wh * offset_scale, C = 2 points at xy + offsets / (W_l, H_l). "
        "Softmax over all P points per (batch, query, head), then bilinear sampling with "
        "zero padding (align_corners=False).",
        required("spatial_shapes", &C::spatialShapes)
            .check(
                [](const std::vector<std::array<uint32_t, 2>>& shapes) { return !shapes.empty(); }, "must not be empty")
            .check(
                [](const std::vector<std::array<uint32_t, 2>>& shapes) {
                    return std::all_of(shapes.begin(), shapes.end(), [](const std::array<uint32_t, 2>& s) {
                        return s[0] >= 1u && s[1] >= 1u && s[0] <= uint32_t(INT32_MAX) && s[1] <= uint32_t(INT32_MAX);
                    });
                },
                "must hold (height, width) pairs >= 1"),
        required("point_counts", &C::pointCounts)
            .check(
                [](const std::vector<uint32_t>& counts, const C& c) { return counts.size() == c.spatialShapes.size(); },
                "must have one count per spatial shape")
            .check(
                [](const std::vector<uint32_t>& counts) {
                    return std::all_of(counts.begin(), counts.end(), [](uint32_t n) { return n >= 1u; });
                },
                "must be >= 1"),
        field("offset_scale", &C::offsetScale).check([](float s) { return std::isfinite(s); }, "must be finite"));
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindDeformableAttention);

}   // namespace pyvai
