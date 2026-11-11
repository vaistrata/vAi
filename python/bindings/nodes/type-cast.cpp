#include "nodes.h"
#include "vai/nodes/type-cast.h"
#include <optional>

namespace pyvai {

void bindTypeCast(nb::module_& m)
{
    nb::class_<vai::node::TypeCast, vai::Node>(m, "TypeCastNode")
        .def(
            "__init__",
            [](vai::node::TypeCast* self, vai::Dtype o_dtype) {
                requireCurrentNet("TypeCastNode");
                if (o_dtype == vai::Dtype::None)
                    throw nb::value_error("TypeCastNode: o_dtype is required");
                new (self) vai::node::TypeCast(vai::node::TypeCast::Config{
                    .oDtype = o_dtype,
                });
            },
            nb::arg("o_dtype"),
            "Element-wise cast of in0 to o_dtype, between fp32 and fp16. When in0 "
            "already has o_dtype, out0 aliases in0's buffer.");
}

}   // namespace pyvai
