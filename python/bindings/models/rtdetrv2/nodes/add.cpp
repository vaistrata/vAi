#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/add.h"

namespace pyvai {

static void bindAdd(nb::module_& m)
{
    using C = model::experimental::rtdetrv2::node::Add::Config;
    bindNode<model::experimental::rtdetrv2::node::Add>(
        m,
        "AddNode",
        "RT-DETRv2 residual add. out0 = in0 + in1; in0 and in1 share shape and dtype. "
        "With constant_in1, in1 is a parameter slot set before the first forward.",
        field("constant_in1", &C::constantIn1));
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindAdd);

}   // namespace pyvai
