#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/rtdetrv2/nodes/query-score.h"

namespace pyvai {

static void bindQueryScore(nb::module_& m)
{
    bindNode<model::experimental::rtdetrv2::node::QueryScore>(
        m,
        "QueryScoreNode",
        "RT-DETRv2 query score. in0 [N, C] fp32 class logits; out0 [N] fp32 holds each "
        "row's maximum.");
}
PYVAI_MODEL_NODE_BINDING(rtdetrv2, bindQueryScore);

}   // namespace pyvai
