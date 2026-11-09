#include "python/bindings/models/yolox/nodes/nodes.h"
#include "models/yolox/nodes/focus-conv.h"
#include <cstdint>

namespace pyvai::yolox_nodes {

void bindFocusConv(nb::module_& m)
{
    nb::class_<yolox::FocusConvNode, vai::Node>(m, "FocusConvNode")
        .def(
            "__init__",
            [](yolox::FocusConvNode* self, uint32_t in_channels, uint32_t out_channels) {
                requireCurrentNet("FocusConvNode");
                if (in_channels == 0u || out_channels == 0u)
                    throw nb::value_error("FocusConvNode: channels must be >= 1");
                // Weights and output take the net's defaults; the core asserts
                // that fp32 weights never write an fp16 output.
                const vai::NeuralNetConfig& net = vai::NeuralNet::current()->config();
                if (net.wDefaultDtype != vai::Dtype::Fp16 && net.oDefaultDtype == vai::Dtype::Fp16)
                    throw nb::value_error("FocusConvNode: fp32 weights cannot write an fp16 output "
                                          "(the net's w_default_dtype / o_default_dtype)");
                new (self) yolox::FocusConvNode(in_channels, out_channels);
            },
            nb::arg("in_channels"), nb::arg("out_channels"),
            "YOLOX Focus folded into its stem Conv-BN-SiLU. in0 [Ci, 2H, 2W] is split into "
            "its four pixel phases (top-left, bottom-left, top-right, bottom-right) stacked "
            "to [4Ci, H, W]; out0 [Co, H, W] = silu(scale * conv3x3 + bias), same padding. "
            "Slots weight [Co, 4Ci, 3, 3], scale and bias [Co].");
}

}   // namespace pyvai::yolox_nodes
