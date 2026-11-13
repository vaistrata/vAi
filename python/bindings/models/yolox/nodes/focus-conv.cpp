#include "python/bindings/nodes/bind-node.h"
#include "models/experimental/yolox/nodes/focus-conv.h"
#include <cstdint>

namespace pyvai {

static void bindFocusConv(nb::module_& m)
{
    nb::class_<model::experimental::yolox::node::FocusConv, vai::Node>(m, "FocusConvNode")
        .def(
            "__init__",
            [](model::experimental::yolox::node::FocusConv* self, uint32_t in_channels, uint32_t out_channels) {
                requireCurrentNet("FocusConvNode");
                if (in_channels == 0u || out_channels == 0u)
                    throw nb::value_error("FocusConvNode: channels must be >= 1");
                new (self) model::experimental::yolox::node::FocusConv(in_channels, out_channels);
            },
            nb::arg("in_channels"),
            nb::arg("out_channels"),
            "YOLOX Focus folded into its stem Conv-BN-SiLU. in0 [Ci, 2H, 2W] is split into "
            "its four pixel phases (top-left, bottom-left, top-right, bottom-right) stacked "
            "to [4Ci, H, W]; out0 [Co, H, W] = silu(scale * conv3x3 + bias), same padding. "
            "Slots weight [Co, 4Ci, 3, 3], scale and bias [Co].");
}
PYVAI_MODEL_NODE_BINDING(yolox, bindFocusConv);

}   // namespace pyvai
