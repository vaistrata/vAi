#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/gated-mul.h"

namespace pyvai {

static void bindGatedMul(nb::module_& m)
{
    using C = vai::node::GatedMul::Config;
    bindNode<vai::node::GatedMul>(m,
                                  "GatedMulNode",
                                  "Gated FFN tail: out0 [..., H] = act(gate) * up, where gate and up are "
                                  "the first and second halves of in0 [..., 2H].",
                                  field("activation", &C::activation));
}
PYVAI_NODE_BINDING(bindGatedMul);

}   // namespace pyvai
