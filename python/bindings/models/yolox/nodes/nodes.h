#ifndef PYVAI_MODELS_YOLOX_NODES_NODES_H
#define PYVAI_MODELS_YOLOX_NODES_NODES_H

#include "python/bindings/nodes/nodes.h"   // requireCurrentNet

namespace pyvai::yolox_nodes {

// One per models/yolox/nodes/<node>.h, in bindYOLOXNodes()'s call order. They
// land in pyvai._pyvai.model_nodes.yolox, where python/tests checks them like
// the core nodes; pyvai's public API does not re-export them.
void bindDecode(nb::module_&);
void bindFocusConv(nb::module_&);
void bindSPPF(nb::module_&);

}   // namespace pyvai::yolox_nodes

#endif   // PYVAI_MODELS_YOLOX_NODES_NODES_H
