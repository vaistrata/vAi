#include "python/bindings/models/yolox/nodes/nodes.h"
#include "models/experimental/yolox/nodes/focus-conv.h"
#include <cstdint>
#include <optional>

namespace yolox = model::experimental::yolox;

namespace pyvai::yolox_nodes {

void bindFocusConv(nb::module_& m)
{
    nb::class_<yolox::node::FocusConv, vai::Node>(m, "FocusConvNode")
        .def(
            "__init__",
            [](yolox::node::FocusConv* self, uint32_t in_channels, uint32_t out_channels,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("FocusConvNode");
                if (in_channels == 0u || out_channels == 0u)
                    throw nb::value_error("FocusConvNode: channels must be >= 1");
                new (self) yolox::node::FocusConv(in_channels, out_channels);
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("in_channels"), nb::arg("out_channels"), nb::arg("o_dtype") = nb::none(),
            "YOLOX Focus folded into its stem Conv-BN-SiLU. in0 [Ci, 2H, 2W] is split into "
            "its four pixel phases (top-left, bottom-left, top-right, bottom-right) stacked "
            "to [4Ci, H, W]; out0 [Co, H, W] = silu(scale * conv3x3 + bias), same padding. "
            "Slots weight [Co, 4Ci, 3, 3], scale and bias [Co]. fp16 activations require "
            "fp16 weights, and so does an fp16 output.");
}

}   // namespace pyvai::yolox_nodes
