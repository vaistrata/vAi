#include "nodes.h"
#include "vai/nodes/ele-affine.h"
#include <optional>

namespace pyvai {

void bindEleAffine(nb::module_& m)
{
    nb::class_<vai::node::EleAffine, vai::Node>(m, "EleAffineNode")
        .def(
            "__init__",
            [](vai::node::EleAffine* self, bool scale_as_input, bool shift_as_input,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("EleAffineNode");
                new (self) vai::node::EleAffine(vai::node::EleAffine::Config{
                    .scaleAsInput = scale_as_input,
                    .shiftAsInput = shift_as_input,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("scale_as_input") = false, nb::arg("shift_as_input") = false, nb::arg("o_dtype") = nb::none(),
            "Element-wise out0 = scale * in0 + shift. The `scale` / `shift` slots are "
            "parameters, or inputs with scale_as_input / shift_as_input True; each "
            "applies when filled or wired, and at least one has to be. Each broadcasts to in0's shape "
            "right-aligned, as long as the axes it keeps are adjacent ([D] or [1, D] "
            "over [T, D], [C, 1, 1] over [C, H, W]). A gated residual x + g * y is "
            "in0=y, scale=g, shift=x.");
}

}   // namespace pyvai
