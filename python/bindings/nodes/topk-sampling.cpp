#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/topk-sampling.h"
#include <cstdint>
#include <string>

namespace pyvai {

static void bindTopKSampling(nb::module_& m)
{
    using C = vai::node::TopKSampling::Config;
    constexpr uint32_t maxK = vai::node::TopKSampling::maxK;   // for the message
    bindNode<vai::node::TopKSampling>(
        m, "TopKSamplingNode",
        "Top-k sampling over fp32 or fp16 logits [..., vocab] (rank >= 2), k <= 128. out0 [...] holds one "
        "raw uint32 index per row: read it with Tensor.to_u32().",
        field("k", &C::k)
            .check([](uint32_t k) { return k >= 1 && k <= vai::node::TopKSampling::maxK; },
                   "must be in [1, " + std::to_string(maxK) + "]"),
        field("temperature", &C::temperature))
        .def("set_uniform", &vai::node::TopKSampling::setUniform, nb::arg("u"),
             "Set the next forward's uniform draw in [0, 1). Call it before every "
             "forward, or every step samples identically.");
}
PYVAI_NODE_BINDING(bindTopKSampling);

}   // namespace pyvai
