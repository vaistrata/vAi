#include "nodes.h"
#include "vai/nodes/gated-mul.h"
#include <optional>

namespace pyvai {

void bindGatedMul(nb::module_& m)
{
    nb::class_<vai::node::GatedMul, vai::Node>(m, "GatedMulNode")
        .def(
            "__init__",
            [](vai::node::GatedMul* self, vai::Epilogue activation, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("GatedMulNode");
                new (self) vai::node::GatedMul(vai::node::GatedMul::Config{
                    .activation = activation,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("activation") = vai::Epilogue::Silu, nb::arg("o_dtype") = nb::none(),
            "Gated FFN tail: out0 [..., H] = act(gate) * up, where gate and up are "
            "the first and second halves of in0 [..., 2H].");
}

}   // namespace pyvai
