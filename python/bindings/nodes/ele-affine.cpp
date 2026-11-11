#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/ele-affine.h"

namespace pyvai {

static void bindEleAffine(nb::module_& m)
{
    using C = vai::node::EleAffine::Config;
    bindNode<vai::node::EleAffine>(m, "EleAffineNode",
                                   "Element-wise out0 = scale * in0 + shift. has_scale / has_shift open the "
                                   "`scale` / `shift` slots: inputs, or parameters at w_dtype with "
                                   "scale_as_input / shift_as_input False. Each broadcasts to in0's shape "
                                   "right-aligned, as long as the axes it keeps are adjacent ([D] or [1, D] "
                                   "over [T, D], [C, 1, 1] over [C, H, W]). A gated residual x + g * y is "
                                   "in0=y, scale=g, shift=x.",
                                   field("has_scale", &C::hasScale),
                                   field("has_shift", &C::hasShift)
                                       .check([](bool shift, const C& c) { return shift || c.hasScale; },
                                              "cannot be False when has_scale is False"),
                                   field("scale_as_input", &C::scaleAsInput), field("shift_as_input", &C::shiftAsInput),
                                   field("w_dtype", &C::wDtype), field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindEleAffine);

}   // namespace pyvai
