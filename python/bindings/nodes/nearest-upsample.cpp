#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/nearest-upsample.h"
#include <cstdint>

namespace pyvai {

static void bindNearestUpsample(nb::module_& m)
{
    nb::class_<vai::node::NearestUpsample, vai::Node>(m, "NearestUpsampleNode")
        .def(
            "__init__",
            [](vai::node::NearestUpsample* self, uint32_t scale) {
                requireCurrentNet("NearestUpsampleNode");
                if (scale <= 1u)
                    throw nb::value_error("NearestUpsampleNode: scale must be >= 2");
                new (self) vai::node::NearestUpsample(scale);
            },
            nb::arg("scale") = 2u,
            "Replicate every [C, H, W] pixel into a scale x scale block (scale >= 2).");
}
PYVAI_NODE_BINDING(bindNearestUpsample);

}   // namespace pyvai
