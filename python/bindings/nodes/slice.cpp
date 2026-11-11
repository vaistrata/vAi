#include "nodes.h"
#include "vai/nodes/slice.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindSlice(nb::module_& m)
{
    nb::class_<vai::node::Slice, vai::Node>(m, "SliceNode")
        .def(
            "__init__",
            [](vai::node::Slice* self, int64_t start, uint32_t count, uint32_t axis,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("SliceNode");
                if (count < 1)
                    throw nb::value_error("SliceNode: count must be >= 1");
                new (self) vai::node::Slice(vai::node::Slice::Config{
                    .axis = axis,
                    .start = start,
                    .count = count,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("start"), nb::arg("count") = 1u, nb::arg("axis") = 0u, nb::arg("o_dtype") = nb::none(),
            "Take `count` consecutive indices from `axis` of in0 (rank >= 2), "
            "starting at `start`; a negative start counts from the end. When only "
            "size-1 axes sit ahead of `axis` (axis 0 always), out0 may alias in0's "
            "buffer and o_dtype (None: the net's o_default_dtype) must be in0's dtype.");
}

}   // namespace pyvai
