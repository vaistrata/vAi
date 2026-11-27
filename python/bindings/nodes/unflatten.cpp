#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/unflatten.h"
#include <cstdint>
#include <vector>

namespace pyvai {

static void bindUnflatten(nb::module_& m)
{
    nb::class_<vai::node::Unflatten, vai::Node>(m, "UnflattenNode")
        .def(
            "__init__",
            [](vai::node::Unflatten* self, int axis, std::vector<int64_t> sizes) {
                requireCurrentNet("UnflattenNode");
                // The core checks sizes only in prepare(): a second -1 or any other
                // non-positive size asserts at the first forward.
                if (sizes.empty())
                    throw nb::value_error("UnflattenNode: sizes must not be empty");
                int inferred = 0;
                for (int64_t s : sizes)
                {
                    if (s == -1)
                        ++inferred;
                    else if (s <= 0)
                        throw nb::value_error("UnflattenNode: sizes must be positive or -1");
                }
                if (inferred > 1)
                    throw nb::value_error("UnflattenNode: at most one size may be -1");
                new (self) vai::node::Unflatten({ .axis = axis, .sizes = std::move(sizes) });
            },
            nb::arg("axis"),
            nb::arg("sizes"),
            "torch.unflatten: `axis` of in0 becomes `sizes`, the other axes pass through. "
            "A negative axis counts from the end. Each size is positive, or -1 for the "
            "leftover, at most once; they must multiply to the axis. out0 aliases in0's "
            "buffer.");
}
PYVAI_NODE_BINDING(bindUnflatten);

}   // namespace pyvai
