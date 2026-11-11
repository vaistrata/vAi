#include "nodes.h"
#include "vai/nodes/convolution.h"
#include <cstdint>
#include <optional>

namespace pyvai {

void bindConvolution(nb::module_& m)
{
    nb::class_<vai::node::Convolution, vai::Node>(m, "ConvolutionNode")
        .def(
            "__init__",
            [](vai::node::Convolution* self, uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size,
               uint32_t stride, bool accept_residual, vai::ResidualOrder residual_order, vai::Epilogue epilogue,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("ConvolutionNode");
                if (in_channels == 0 || out_channels == 0)
                    throw nb::value_error("ConvolutionNode: channels must be >= 1");
                if (kernel_size == 0 || kernel_size % 2 == 0)
                    throw nb::value_error("ConvolutionNode: kernel_size must be odd");
                if (stride == 0)
                    throw nb::value_error("ConvolutionNode: stride must be >= 1");
                new (self) vai::node::Convolution(vai::node::Convolution::Config{
                    .inChannels = in_channels,
                    .outChannels = out_channels,
                    .kernelSize = kernel_size,
                    .stride = stride,
                    .acceptResidual = accept_residual,
                    .residualOrder = residual_order,
                    .epilogue = epilogue,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("in_channels"), nb::arg("out_channels"), nb::arg("kernel_size"), nb::arg("stride"),
            nb::arg("accept_residual") = false, nb::arg("residual_order") = vai::ResidualOrder::AfterEpilogue,
            nb::arg("epilogue") = vai::Epilogue::None, nb::arg("o_dtype") = nb::none(),
            "The `scale` / `bias` slots [out_channels] apply when filled, as "
            "out0 = scale * conv + bias; a batch norm folds into them. "
            "accept_residual opens the `residual` input [out_channels, Ho, Wo] (input "
            "dtype), added after the epilogue or before it per residual_order. fp16 "
            "activations require fp16 weights, and so does an fp16 output.");
}

}   // namespace pyvai
