#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/slice.h"
#include <cstdint>

namespace pyvai {

static void bindSlice(nb::module_& m)
{
    using C = vai::node::Slice::Config;
    bindNode<vai::node::Slice>(m, "SliceNode",
                               "Take `count` consecutive indices from `axis` of in0 (rank >= 2), "
                               "starting at `start`; a negative start counts from the end. When only "
                               "size-1 axes sit ahead of `axis` (axis 0 always), out0 may alias in0's "
                               "buffer and o_dtype (None: the net's o_default_dtype) must be in0's dtype.",
                               required("start", &C::start),
                               field("count", &C::count).check(atLeast(1u), "must be >= 1"), field("axis", &C::axis),
                               field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindSlice);

}   // namespace pyvai
