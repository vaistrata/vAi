#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/mul.h"

namespace pyvai {

static void bindMul(nb::module_& m)
{
    using C = vai::node::Mul::Config;
    bindNode<vai::node::Mul>(m, "MulNode",
                             "Element-wise out0 = in0 * in1. Both inputs share one shape and "
                             "dtype; there is no broadcasting.",
                             field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindMul);

}   // namespace pyvai
