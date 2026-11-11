#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/argmax.h"

namespace pyvai {

static void bindArgmax(nb::module_& m)
{
    bindNode<vai::node::Argmax>(m, "ArgmaxNode",
                                "Greedy pick over [M, N] logits, fp32 or fp16. out0 [M] holds raw uint32 "
                                "indices: read it with Tensor.to_u32(), not to_numpy().");
}
PYVAI_NODE_BINDING(bindArgmax);

}   // namespace pyvai
