#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/layer-norm.h"

namespace pyvai {

static void bindLayerNorm(nb::module_& m)
{
    using C = vai::node::LayerNorm::Config;
    bindNode<vai::node::LayerNorm>(m, "LayerNormNode",
                                   "Normalize an [..., feature] input (rank >= 2) over its last axis: "
                                   "(x - mean) * rsqrt(var + eps) * gamma + beta. has_beta=False drops the "
                                   "`beta` slot. Row statistics accumulate in fp32 whatever the I/O dtypes.",
                                   field("eps", &C::eps), field("has_beta", &C::hasBeta), field("w_dtype", &C::wDtype),
                                   field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindLayerNorm);

}   // namespace pyvai
