#include "nodes.h"
#include "vai/nodes/row-slice-scatter.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace pyvai {

void bindRowSliceScatter(nb::module_& m)
{
    nb::class_<vai::node::RowSliceScatter, vai::Node>(m, "RowSliceScatterNode")
        .def(
            "__init__",
            [](vai::node::RowSliceScatter* self, uint32_t slice_size) {
                requireCurrentNet("RowSliceScatterNode");
                new (self) vai::node::RowSliceScatter(vai::node::RowSliceScatter::Config{
                    .sliceSize = slice_size,
                });
            },
            nb::arg("slice_size"),
            "torch.slice_scatter on dim 0: out0 is in0 [T, D] with rows "
            "[row_starts[b], row_starts[b] + slice_size) taken from slice b of in1, "
            "[B, slice_size, D] or [slice_size, D] for one. Both inputs share a dtype. A "
            "forward without in1 passes in0 through; slice_size 0 says in1 is never fed.")
        .def(
            "set_row_starts",
            [](vai::node::RowSliceScatter& self, std::vector<uint32_t> rows) { self.setRowStarts(std::move(rows)); },
            nb::arg("rows"),
            "The first row of each slice of in1; the default is one slice at row 0. A "
            "change re-records the net. A count other than in1's, or a slice past the "
            "end of in0, is the next forward's error.");
}

}   // namespace pyvai
