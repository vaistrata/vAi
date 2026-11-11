#include "python/bindings/nodes/bind-node.h"
#include "models/yolox/nodes/sppf.h"

namespace pyvai {

static void bindSPPF(nb::module_& m)
{
    using C = yolox::SPPFNode::Config;
    bindNode<yolox::SPPFNode>(m, "SPPFNode",
                              "YOLOX SPP. in0 [C, H, W] with H * W <= 1024; out0 [4C, H, W] stacks in0, p(in0), "
                              "p(p(in0)) and p(p(p(in0))) on channels, p a 5x5 stride-1 max pool that ignores "
                              "samples outside the input.",
                              field("o_dtype", &C::oDtype));
}
PYVAI_MODEL_NODE_BINDING(yolox, bindSPPF);

}   // namespace pyvai
