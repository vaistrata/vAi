#include "nodes.h"
#include "vai/nodes/cyclic-table-reader.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindCyclicTableReader(nb::module_& m)
{
    nb::class_<vai::node::CyclicTableReader, vai::Node>(m, "CyclicTableReaderNode")
        .def(
            "__init__",
            [](vai::node::CyclicTableReader* self, uint32_t num_rows, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("CyclicTableReaderNode");
                if (num_rows < 1)
                    throw nb::value_error("CyclicTableReaderNode: num_rows must be >= 1");
                new (self) vai::node::CyclicTableReader(vai::node::CyclicTableReader::Config{
                    .numRows = num_rows,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("num_rows"), nb::arg("o_dtype") = nb::none(),
            "Reads one row of the `table` parameter [num_rows, ...] per forward: forward k after rewind() writes row k "
            "% num_rows "
            "to out0, the table's shape without its leading axis. in0 only schedules "
            "the node; what it holds is never read.")
        .def("rewind", &vai::node::CyclicTableReader::rewind,
             "Start over at row 0. Not synchronized: call net.sync() first if a "
             "forward may still be running.");
}

}   // namespace pyvai
