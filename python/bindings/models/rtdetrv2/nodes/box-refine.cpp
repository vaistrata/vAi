#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/box-refine.h"

namespace pyvai {

static void bindBoxRefine(nb::module_& m)
{
    bindNode<model::experimental::rtdetrv2::node::BoxRefine>(
        m,
        "BoxRefineNode",
        "RT-DETRv2 box refinement. Inputs boxes and delta [Q, 4] fp32; out0 [Q, 4] fp32 = "
        "sigmoid(delta + inverse_sigmoid(boxes)), boxes clamped to [0, 1] and both "
        "inverse_sigmoid terms to at least 1e-5.");
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindBoxRefine);

}   // namespace pyvai
