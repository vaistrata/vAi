#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/rms-norm.h"

namespace pyvai {

static void bindRmsNorm(nb::module_& m)
{
    using C = vai::node::RmsNorm::Config;
    bindNode<vai::node::RmsNorm>(m, "RmsNormNode",
                                 "RMS normalization of [..., feature] over the feature axis: x * "
                                 "rsqrt(mean(x^2) + eps) * scale + shift. has_scale / has_shift open the "
                                 "`scale` / `shift` slots [feature]: parameters at w_dtype (None = F32), "
                                 "or inputs computed upstream with coeffs_as_inputs=True. Row statistics "
                                 "accumulate in fp32 whatever the I/O dtypes.",
                                 field("eps", &C::eps), field("has_scale", &C::hasScale),
                                 field("has_shift", &C::hasShift), field("coeffs_as_inputs", &C::coeffsAsInputs),
                                 field("w_dtype", &C::wDtype), field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindRmsNorm);

}   // namespace pyvai
