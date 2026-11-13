#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/slice.h"
#include <cstdint>

namespace pyvai {

static void bindSlice(nb::module_& m)
{
    using C = vai::node::Slice::Config;
    bindNode<vai::node::Slice>(m,
                               "SliceNode",
                               "Take `count` consecutive indices from `axis` of in0 (rank >= 2), "
                               "starting at `start`; a negative start counts from the end. When only "
                               "size-1 axes sit ahead of `axis` (axis 0 always), out0 may alias in0's "
                               "buffer, so in0 must be at the net's o_default_dtype.",
                               required("start", &C::start),
                               field("count", &C::count).check(atLeast(1u), "must be >= 1"),
                               field("axis", &C::axis));
}
PYVAI_NODE_BINDING(bindSlice);

}   // namespace pyvai
