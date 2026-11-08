#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/convolution.h"
#include <cstdint>

namespace pyvai {

static void bindConvolution(nb::module_& m)
{
    using C = vai::node::Convolution::Config;
    bindNode<vai::node::Convolution>(
        m, "ConvolutionNode",
        "has_bias / has_scale add `bias` / `scale` slots [out_channels], applied "
        "as out0 = scale * conv + bias; a batch norm folds into them. fp16 "
        "activations require fp16 weights, and so does an fp16 output.",
        required("in_channels", &C::inChannels).check(atLeast(1u), "must be >= 1"),
        required("out_channels", &C::outChannels).check(atLeast(1u), "must be >= 1"),
        required("kernel_size", &C::kernelSize).check([](uint32_t k) { return k % 2 == 1; }, "must be odd"),
        required("stride", &C::stride).check(atLeast(1u), "must be >= 1"), field("has_bias", &C::hasBias),
        field("has_scale", &C::hasScale), field("epilogue", &C::epilogue), field("w_dtype", &C::wDtype),
        // The core resolves None to the net's defaults, then asserts that fp32
        // weights never write an fp16 output.
        field("o_dtype", &C::oDtype)
            .check(
                [](vai::Dtype o, const C& c) {
                    const vai::NeuralNetConfig& net = vai::NeuralNet::current()->config();
                    const vai::Dtype w = c.wDtype == vai::Dtype::None ? net.wDefaultDtype : c.wDtype;
                    return w == vai::Dtype::Fp16 || (o == vai::Dtype::None ? net.oDefaultDtype : o) != vai::Dtype::Fp16;
                },
                "cannot be fp16 with fp32 weights (w_dtype / o_dtype, or the net's defaults)"))
        .def("enable_residual_add", &vai::node::Convolution::enableResidualAdd,
             "Fuse a residual add through a new `residual` input slot [out_channels, "
             "Ho, Wo]; call before wiring it. Returns False and changes nothing unless "
             "the conv is 3x3, stride 1 and scaled with in_channels == out_channels.");
}
PYVAI_NODE_BINDING(bindConvolution);

}   // namespace pyvai
