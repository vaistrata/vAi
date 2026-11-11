#include "nodes.h"
#include "vai/nodes/topk-sampling.h"
#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

void bindTopKSampling(nb::module_& m)
{
    nb::class_<vai::node::TopKSampling, vai::Node>(m, "TopKSamplingNode")
        .def(
            "__init__",
            [](vai::node::TopKSampling* self, uint32_t k, float temperature) {
                requireCurrentNet("TopKSamplingNode");
                if (k < 1 || k > vai::node::TopKSampling::maxK)
                    throw nb::value_error(
                        ("TopKSamplingNode: k must be in [1, " + std::to_string(vai::node::TopKSampling::maxK) + "]")
                            .c_str());
                new (self) vai::node::TopKSampling(vai::node::TopKSampling::Config{
                    .k = k,
                    .temperature = temperature,
                });
            },
            nb::arg("k") = 40u, nb::arg("temperature") = 1.0f,
            "Top-k sampling over fp32 or fp16 logits [..., vocab] (rank >= 2), k <= 128. out0 [...] holds one "
            "raw uint32 index per row: read it with Tensor.to_u32().")
        .def("set_uniform", &vai::node::TopKSampling::setUniform, nb::arg("u"),
             "Set the next forward's uniform draw in [0, 1). Call it before every "
             "forward, or every step samples identically.");
}

}   // namespace pyvai
