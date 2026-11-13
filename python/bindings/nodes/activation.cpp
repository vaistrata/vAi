#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/activation.h"

namespace pyvai {

static void bindActivation(nb::module_& m)
{
    // `epilogue` is required (C++ defaults it to None): a standalone no-op activation
    // is almost surely a bug.
    using C = vai::node::Activation::Config;
    bindNode<vai::node::Activation>(m,
                                    "ActivationNode",
                                    "Element-wise epilogue over in0. in_place=True aliases out0 onto in0's "
                                    "buffer, so in0 must be at the net's o_default_dtype.",
                                    required("epilogue", &C::epilogue),
                                    field("in_place", &C::inPlace));
}
PYVAI_NODE_BINDING(bindActivation);

}   // namespace pyvai
