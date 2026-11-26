#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/index-select.h"

namespace pyvai {

static void bindIndexSelect(nb::module_& m)
{
    using C = model::experimental::rtdetrv2::node::IndexSelect::Config;
    bindNode<model::experimental::rtdetrv2::node::IndexSelect>(
        m,
        "IndexSelectNode",
        "RT-DETRv2 row gather. in0 [N, d1, ..., dn] fp32 or fp16, indices [K] raw uint32 "
        "in 4-byte fp32 slots (as TopKNode writes them); out0 [K, d1, ..., dn] = in0[indices], "
        "in0's dtype. An index >= N gives a zero row. With constant_input, in0 is a parameter "
        "slot set before the first forward.",
        field("constant_input", &C::constantInput));
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindIndexSelect);

}   // namespace pyvai
