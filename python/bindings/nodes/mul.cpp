#include "nodes.h"
#include "vai/nodes/mul.h"
#include <optional>

namespace pyvai {

void bindMul(nb::module_& m)
{
    nb::class_<vai::node::Mul, vai::Node>(m, "MulNode")
        .def(
            "__init__",
            [](vai::node::Mul* self, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("MulNode");
                new (self) vai::node::Mul(vai::node::Mul::Config{});
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("o_dtype") = nb::none(),
            "Element-wise out0 = in0 * in1. Both inputs share one shape and "
            "dtype; there is no broadcasting.");
}

}   // namespace pyvai
