#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/rms-norm.h"

namespace pyvai {

static void bindRmsNorm(nb::module_& m)
{
    using C = vai::node::RmsNorm::Config;
    bindNode<vai::node::RmsNorm>(m,
                                 "RmsNormNode",
                                 "RMS normalization of [..., feature] over the feature axis: x * "
                                 "rsqrt(mean(x^2) + eps) * scale + shift. `scale` and `shift` are [feature] "
                                 "parameters, or inputs computed upstream with coeffs_as_inputs=True; each "
                                 "applies when it is set or wired. accept_residual opens a `residual` input, "
                                 "added to the output. Row statistics accumulate in fp32 whatever the I/O "
                                 "dtypes.",
                                 field("eps", &C::eps),
                                 field("coeffs_as_inputs", &C::coeffsAsInputs),
                                 field("accept_residual", &C::acceptResidual));
}
PYVAI_NODE_BINDING(bindRmsNorm);

}   // namespace pyvai
