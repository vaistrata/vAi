#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/topk.h"
#include <cstdint>

namespace pyvai {

static void bindTopK(nb::module_& m)
{
    using C = model::experimental::rtdetrv2::node::TopK::Config;
    bindNode<model::experimental::rtdetrv2::node::TopK>(
        m,
        "TopKNode",
        "RT-DETRv2 row-wise top-k. in0 [N] or [M, N] fp32 or fp16 with N >= k; outputs "
        "values [k] or [M, k] in in0's dtype and indices, raw uint32 positions in 4-byte "
        "fp32 slots. Rows sort by descending value, ties by ascending index, NaN last.",
        field("k", &C::k).check([](uint32_t k) { return k >= 1u && k <= 512u; }, "must be in [1, 512]"));
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindTopK);

}   // namespace pyvai
