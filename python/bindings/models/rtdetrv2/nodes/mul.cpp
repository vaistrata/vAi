#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/mul.h"

namespace pyvai {

static void bindMul(nb::module_& m)
{
    using C = model::experimental::rtdetrv2::node::Mul::Config;
    bindNode<model::experimental::rtdetrv2::node::Mul>(
        m,
        "MulNode",
        "RT-DETRv2 element-wise multiply with broadcasting. out0 = in0 * in1; axes align "
        "from the right and each pair must match or be 1. in0 and in1 share a dtype. "
        "With constant_in1, in1 is a parameter slot set before the first forward.",
        field("constant_in1", &C::constantIn1));
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindMul);

}   // namespace pyvai
