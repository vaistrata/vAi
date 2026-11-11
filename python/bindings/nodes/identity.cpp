#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/identity.h"

namespace pyvai {

static void bindIdentity(nb::module_& m)
{
    bindNode<vai::node::Identity>(m, "IdentityNode",
                                  "Pass-through: out0 aliases in0's buffer, any shape and dtype. It makes a "
                                  "named junction: a group's in0, or one tensor feeding several consumers.");
}
PYVAI_NODE_BINDING(bindIdentity);

}   // namespace pyvai
