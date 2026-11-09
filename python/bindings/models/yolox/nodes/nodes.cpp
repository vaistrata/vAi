#include "python/bindings/models/models.h"
#include "python/bindings/models/yolox/nodes/nodes.h"

namespace pyvai {

// Call order must match the list in nodes.h.
void bindYOLOXNodes(nb::module_& m)
{
    yolox_nodes::bindDecode(m);
    yolox_nodes::bindFocusConv(m);
    yolox_nodes::bindSPPF(m);
}

}   // namespace pyvai
