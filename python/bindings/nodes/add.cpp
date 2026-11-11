#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/add.h"

namespace pyvai {

static void bindAdd(nb::module_& m)
{
    using C = vai::node::Add::Config;
    bindNode<vai::node::Add>(m, "AddNode",
                             "Element-wise out0 = in0 + in1. Both inputs share one shape and "
                             "dtype; there is no broadcasting.",
                             field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindAdd);

}   // namespace pyvai
