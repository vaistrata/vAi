#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/concat.h"
#include <cstdint>

namespace pyvai {

static void bindConcat(nb::module_& m)
{
    using C = vai::node::Concat::Config;
    bindNode<vai::node::Concat>(m,
                                "ConcatNode",
                                "Concatenate in0..inN (one shared dtype) along `axis` (negative counts "
                                "from the end). When only size-1 axes sit ahead of `axis` and the inputs are "
                                "at the net's o_default_dtype, buffer copies place them and out0 aliases the "
                                "input when only one arrives; otherwise a gather kernel joins at most 8 "
                                "arriving inputs. "
                                "allow_partial_inputs lets a forward bring only some of the inputs.",
                                field("axis", &C::axis),
                                field("input_count", &C::inputCount).check(atLeast(2u), "must be >= 2"),
                                field("allow_partial_inputs", &C::allowPartialInputs));
}
PYVAI_NODE_BINDING(bindConcat);

}   // namespace pyvai
