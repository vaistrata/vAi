#include "nodes.h"
#include "vai/nodes/gate.h"

namespace pyvai {

void bindGate(nb::module_& m)
{
    nb::class_<vai::node::Gate, vai::Node>(m, "GateNode")
        .def(
            "__init__",
            [](vai::node::Gate* self) {
                requireCurrentNet("GateNode");
                new (self) vai::node::Gate();
            },
            "Conditional branch: in0 leaves on `then` or `else`, both aliasing its "
            "buffer. A node only the closed branch reaches neither prepares nor "
            "runs, though its weights still upload. The branches need not agree on "
            "shape or dtype.")

        .def_prop_rw(
            "open", [](const vai::node::Gate& self) { return self.isOpen(); },
            [](vai::node::Gate& self, bool open) { self.setOpen(open); },
            "Which branch the next forward takes: True is `then`, False is `else`. "
            "Flipping it re-records the net, so set it while the net is idle.");
}

}   // namespace pyvai
