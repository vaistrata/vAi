#include "nodes.h"
#include "vai/nodes/concat.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindConcat(nb::module_& m)
{
    nb::class_<vai::node::Concat, vai::Node>(m, "ConcatNode")
        .def(
            "__init__",
            [](vai::node::Concat* self, int axis, uint32_t input_count, bool allow_partial_inputs,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("ConcatNode");
                if (input_count < 2)
                    throw nb::value_error("ConcatNode: input_count must be >= 2");
                new (self) vai::node::Concat(vai::node::Concat::Config{
                    .axis = axis,
                    .inputCount = input_count,
                    .allowPartialInputs = allow_partial_inputs,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("axis") = 0, nb::arg("input_count") = 2u, nb::arg("allow_partial_inputs") = false,
            nb::arg("o_dtype") = nb::none(),
            "Concatenate in0..inN (one shared dtype) along `axis` (negative counts "
            "from the end). When only size-1 axes sit ahead of `axis` and o_dtype is "
            "the inputs' dtype, buffer copies place the inputs and out0 aliases the "
            "input when only one arrives; otherwise a gather kernel joins at most 8 "
            "arriving inputs. "
            "allow_partial_inputs lets a forward bring only some of the inputs.");
}

}   // namespace pyvai
