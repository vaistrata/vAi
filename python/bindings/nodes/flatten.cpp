#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/flatten.h"

namespace pyvai {

static void bindFlatten(nb::module_& m)
{
    bindNode<vai::node::Flatten>(m, "FlattenNode",
                                 "Collapse in0 of any shape to out0 [N] in memory order, the same as "
                                 "ReshapeNode([-1]). out0 aliases in0's buffer.");
}
PYVAI_NODE_BINDING(bindFlatten);

}   // namespace pyvai
