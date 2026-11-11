#include "nodes.h"
#include "vai/nodes/nearest-upsample.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindNearestUpsample(nb::module_& m)
{
    nb::class_<vai::node::NearestUpsample, vai::Node>(m, "NearestUpsampleNode")
        .def(
            "__init__",
            [](vai::node::NearestUpsample* self,
               uint32_t scale,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("NearestUpsampleNode");
                if (scale <= 1u)
                    throw nb::value_error("NearestUpsampleNode: scale must be >= 2");
                new (self) vai::node::NearestUpsample(scale, vai::node::NearestUpsample::Config{});
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("scale")   = 2u,
            nb::arg("o_dtype") = nb::none(),
            "Replicate every [C, H, W] pixel into a scale x scale block (scale >= 2)."
        );
}

}   // namespace pyvai
