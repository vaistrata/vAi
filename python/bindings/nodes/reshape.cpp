#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/reshape.h"
#include <cstdint>
#include <vector>

namespace pyvai {

static void bindReshape(nb::module_& m)
{
    nb::class_<vai::node::Reshape, vai::Node>(m, "ReshapeNode")
        .def(
            "__init__",
            [](vai::node::Reshape* self, std::vector<int64_t> dims)
            {
                requireCurrentNet("ReshapeNode");
                // The core checks dims only in prepare(): a second -1 or any other
                // negative dim asserts at the first forward.
                int inferred = 0;
                for (int64_t d : dims)
                {
                    if (d == -1) ++inferred;
                    else if (d < 0)
                        throw nb::value_error(
                            "ReshapeNode: dims must be positive, 0 or -1");
                }
                if (inferred > 1)
                    throw nb::value_error(
                        "ReshapeNode: at most one dim may be -1");
                new (self) vai::node::Reshape(std::move(dims));
            },
            nb::arg("dims"),
            "Each dim is a size, 0 for in0's dimension at that position, or -1 for "
            "the leftover, at most once. The resolved dims must multiply to in0's "
            "element count. out0 aliases in0's buffer."
        );
}
PYVAI_NODE_BINDING(bindReshape);

}   // namespace pyvai
