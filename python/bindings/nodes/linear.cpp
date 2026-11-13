#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/linear.h"
#include <cstdint>

namespace pyvai {

static void bindLinear(nb::module_& m)
{
    using C = vai::node::Linear::Config;
    bindNode<vai::node::Linear>(
        m,
        "LinearNode",
        "Dense layer: out0 = scalar_scale * epilogue(in0 W^T + bias) + residual, "
        "or with the residual inside the epilogue per residual_order. `bias` "
        "applies when it is set. accept_residual opens the `residual` input; "
        "bias and residual are each [..., out_dim] or rows that repeat over the "
        "output, such as [out_dim]. split_size is the K length per split-K span "
        "(0 = the kernel's heuristic, otherwise >= 32) and only matters past 8 "
        "input rows.",
        required("in_dim", &C::inDim),
        required("out_dim", &C::outDim),
        field("accept_residual", &C::acceptResidual),
        field("residual_order", &C::residualOrder),
        field("epilogue", &C::epilogue),
        field("scalar_scale", &C::scalarScale),
        // The core asserts (aborts) on the tile path.
        field("split_size", &C::splitSize).check([](uint32_t s) { return s == 0 || s >= 32; }, "must be 0 or >= 32"));
}
PYVAI_NODE_BINDING(bindLinear);

}   // namespace pyvai
