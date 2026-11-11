#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/gate.h"

namespace pyvai {

static void bindGate(nb::module_& m)
{
    bindNode<vai::node::Gate>(m, "GateNode",
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
PYVAI_NODE_BINDING(bindGate);

}   // namespace pyvai
