#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/convolution.h"
#include <cstdint>

namespace pyvai {

static void bindConvolution(nb::module_& m)
{
    using C = vai::node::Convolution::Config;
    bindNode<vai::node::Convolution>(
        m,
        "ConvolutionNode",
        "out0 = epilogue(scale * conv(in0) + bias) over [..., in_channels, H, W], "
        "with same-padding: each spatial output size is ceil(input / stride). "
        "`scale` and `bias` [out_channels] each apply when set; a batch norm folds "
        "into them. accept_residual opens the `residual` input [..., out_channels, "
        "Ho, Wo], added after the epilogue or before it per residual_order.",
        required("in_channels", &C::inChannels).check(atLeast(1u), "must be >= 1"),
        required("out_channels", &C::outChannels).check(atLeast(1u), "must be >= 1"),
        required("kernel_size", &C::kernelSize).check([](uint32_t k) { return k % 2 == 1; }, "must be odd"),
        required("stride", &C::stride).check(atLeast(1u), "must be >= 1"),
        field("accept_residual", &C::acceptResidual),
        field("residual_order", &C::residualOrder),
        field("epilogue", &C::epilogue));
}
PYVAI_NODE_BINDING(bindConvolution);

}   // namespace pyvai
