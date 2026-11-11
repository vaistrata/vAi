#include "nodes.h"
#include "vai/nodes/pool2d.h"
#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

namespace {

template <vai::PoolMode mode>
void bindPool2dMode(nb::module_& m, const char* name, const char* doc)
{
    using Node = vai::node::Pool2d<mode>;

    nb::class_<Node, vai::Node>(m, name).def(
        "__init__",
        [name](Node* self, uint32_t kernel_size, uint32_t stride, uint32_t padding, bool ceil_mode,
               bool count_include_pad, bool channels_last, std::optional<vai::Dtype> o_dtype) {
            requireCurrentNet(name);
            if (kernel_size < 1 || stride < 1)
                throw nb::value_error((std::string(name) + ": kernel_size and stride must be >= 1").c_str());
            if (padding * 2 > kernel_size)
                throw nb::value_error((std::string(name) + ": padding must be at most half of kernel_size").c_str());
            new (self) Node(typename Node::Config{
                .kernelSize = kernel_size,
                .stride = stride,
                .padding = padding,
                .ceilMode = ceil_mode,
                .countIncludePad = count_include_pad,
                .channelsLast = channels_last,
            });
            self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
        },
        nb::arg("kernel_size"), nb::arg("stride"), nb::arg("padding") = 0u, nb::arg("ceil_mode") = false,
        nb::arg("count_include_pad") = true, nb::arg("channels_last") = false, nb::arg("o_dtype") = nb::none(), doc);
}

}   // namespace


void bindPool2d(nb::module_& m)
{
    bindPool2dMode<vai::PoolMode::Max>(m, "MaxPool2dNode",
        "One square window on both spatial axes of [..., C, H, W], or [..., H, W, C] "
        "with channels_last: Ho = (H + 2*padding - kernel_size) / stride + 1, rounded "
        "up with ceil_mode, and Wo likewise. Samples outside the input are ignored.");
    bindPool2dMode<vai::PoolMode::Avg>(m, "AvgPool2dNode",
        "One square window on both spatial axes of [..., C, H, W], or [..., H, W, C] "
        "with channels_last: Ho = (H + 2*padding - kernel_size) / stride + 1, rounded "
        "up with ceil_mode, and Wo likewise. Padding adds zeros, which "
        "count_include_pad also counts in the divisor.");
}

}   // namespace pyvai
