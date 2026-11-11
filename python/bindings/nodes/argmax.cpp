#include "nodes.h"
#include "vai/nodes/argmax.h"
#include <optional>

namespace pyvai {

void bindArgmax(nb::module_& m)
{
    nb::class_<vai::node::Argmax, vai::Node>(m, "ArgmaxNode")
        .def(
            "__init__",
            [](vai::node::Argmax* self) {
                requireCurrentNet("ArgmaxNode");
                new (self) vai::node::Argmax(vai::node::Argmax::Config{});
            },
            "Greedy pick over [M, N] logits, fp32 or fp16. out0 [M] holds raw uint32 "
            "indices: read it with Tensor.to_u32(), not to_numpy().");
}

}   // namespace pyvai
