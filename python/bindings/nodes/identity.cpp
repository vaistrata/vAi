#include "nodes.h"
#include "vai/nodes/identity.h"

namespace pyvai {

void bindIdentity(nb::module_& m)
{
    nb::class_<vai::node::Identity, vai::Node>(m, "IdentityNode")
        .def(
            "__init__",
            [](vai::node::Identity* self) {
                requireCurrentNet("IdentityNode");
                new (self) vai::node::Identity();
            },
            "Pass-through: out0 aliases in0's buffer, any shape and dtype. It makes a "
            "named junction: a group's in0, or one tensor feeding several consumers.");
}

}   // namespace pyvai
