#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/ele-affine.h"

namespace pyvai {

static void bindEleAffine(nb::module_& m)
{
    using C = vai::node::EleAffine::Config;
    bindNode<vai::node::EleAffine>(m,
                                   "EleAffineNode",
                                   "Element-wise out0 = scale * in0 + shift. `scale` / `shift` are parameters, or "
                                   "inputs with scale_as_input / shift_as_input. Each applies when it is set "
                                   "or wired (scale 1 and shift 0 otherwise), and at least one has to be. "
                                   "Each broadcasts to in0's shape "
                                   "right-aligned, as long as the axes it keeps are adjacent ([D] or [1, D] "
                                   "over [T, D], [C, 1, 1] over [C, H, W]). A gated residual x + g * y is "
                                   "in0=y, scale=g, shift=x.",
                                   field("scale_as_input", &C::scaleAsInput),
                                   field("shift_as_input", &C::shiftAsInput));
}
PYVAI_NODE_BINDING(bindEleAffine);

}   // namespace pyvai
