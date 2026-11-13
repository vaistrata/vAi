#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/layer-norm.h"

namespace pyvai {

static void bindLayerNorm(nb::module_& m)
{
    using C = vai::node::LayerNorm::Config;
    bindNode<vai::node::LayerNorm>(m,
                                   "LayerNormNode",
                                   "Normalize an [..., feature] input (rank >= 2) over its last axis: "
                                   "(x - mean) * rsqrt(var + eps) * scale + shift. `scale` and `shift` are "
                                   "[feature] parameters, or inputs computed upstream with "
                                   "coeffs_as_inputs=True; shift applies when it is set or wired. "
                                   "accept_residual opens a `residual` input, added to the output. Row "
                                   "statistics accumulate in fp32 whatever the I/O dtypes.",
                                   field("eps", &C::eps),
                                   field("coeffs_as_inputs", &C::coeffsAsInputs),
                                   field("accept_residual", &C::acceptResidual));
}
PYVAI_NODE_BINDING(bindLayerNorm);

}   // namespace pyvai
