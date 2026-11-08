#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/cyclic-table-reader.h"
#include <cstdint>

namespace pyvai {

static void bindCyclicTableReader(nb::module_& m)
{
    using C = vai::node::CyclicTableReader::Config;
    bindNode<vai::node::CyclicTableReader>(m, "CyclicTableReaderNode",
                                           "Reads one row of the `table` parameter [num_rows, ...] (at table_dtype, "
                                           "None = F32) per forward: forward k after rewind() writes row k % num_rows "
                                           "to out0, the table's shape without its leading axis. in0 only schedules "
                                           "the node; what it holds is never read.",
                                           required("num_rows", &C::numRows).check(atLeast(1u), "must be >= 1"),
                                           field("table_dtype", &C::tableDtype), field("o_dtype", &C::oDtype))
        .def("rewind", &vai::node::CyclicTableReader::rewind,
             "Start over at row 0. Not synchronized: call net.sync() first if a "
             "forward may still be running.");
}
PYVAI_NODE_BINDING(bindCyclicTableReader);

}   // namespace pyvai
