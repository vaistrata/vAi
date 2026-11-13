#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/group-norm.h"
#include <cstdint>

namespace pyvai {

static void bindGroupNorm(nb::module_& m)
{
    using C = vai::node::GroupNorm::Config;
    bindNode<vai::node::GroupNorm>(m,
                                   "GroupNormNode",
                                   "Group normalization over [..., C, H, W]: out0 = epilogue(scale[c] * (in0 - "
                                   "mean) * rsqrt(var + eps) + shift[c]) + residual, the statistics taken "
                                   "per sample over each of `groups` consecutive channel groups (C must be a "
                                   "multiple of groups). `scale` and `shift` are [C] parameters, or inputs "
                                   "computed upstream with coeffs_as_inputs=True; shift applies when it is "
                                   "set or wired. accept_residual opens the `residual` input, added after "
                                   "the epilogue or before it per residual_order.",
                                   required("groups", &C::groups).check(atLeast(1u), "must be >= 1"),
                                   field("eps", &C::eps),
                                   field("coeffs_as_inputs", &C::coeffsAsInputs),
                                   field("accept_residual", &C::acceptResidual),
                                   field("residual_order", &C::residualOrder),
                                   field("epilogue", &C::epilogue));
}
PYVAI_NODE_BINDING(bindGroupNorm);

}   // namespace pyvai
