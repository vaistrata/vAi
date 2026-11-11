#include "nodes.h"
#include "vai/nodes/add.h"
#include <optional>

namespace pyvai {

void bindAdd(nb::module_& m)
{
    nb::class_<vai::node::Add, vai::Node>(m, "AddNode")
        .def(
            "__init__",
            [](vai::node::Add* self, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("AddNode");
                new (self) vai::node::Add(vai::node::Add::Config{});
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("o_dtype") = nb::none(),
            "Element-wise out0 = in0 + in1. Both inputs share one shape and "
            "dtype; there is no broadcasting.");
}

}   // namespace pyvai
