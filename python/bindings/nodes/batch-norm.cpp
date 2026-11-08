#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/batch-norm.h"

namespace pyvai {

static void bindBatchNorm(nb::module_& m)
{
    using C = vai::node::BatchNorm::Config;
    bindNode<vai::node::BatchNorm>(m, "BatchNormNode",
                                   "Inference-time batch norm over [channels, H, W]: out0 = gamma[c] * "
                                   "(in0 - mean[c]) * rsqrt(variance[c] + eps) + beta[c]. The `gamma`, "
                                   "`beta`, `mean` and `variance` slots are each [channels] at w_dtype.",
                                   required("channels", &C::channels), field("eps", &C::eps),
                                   field("w_dtype", &C::wDtype), field("o_dtype", &C::oDtype));
}
PYVAI_NODE_BINDING(bindBatchNorm);

}   // namespace pyvai
