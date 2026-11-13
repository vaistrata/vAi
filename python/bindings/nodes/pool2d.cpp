#include "python/bindings/nodes/bind-node.h"
#include "vai/nodes/pool2d.h"
#include <cstdint>

namespace pyvai {

namespace {

template <vai::PoolMode mode>
void bindPool2dMode(nb::module_& m, const char* name, const char* doc)
{
    using Node = vai::node::Pool2d<mode>;
    using C = typename Node::Config;
    bindNode<Node>(
        m,
        name,
        doc,
        required("kernel_size", &C::kernelSize).check(atLeast(1u), "must be >= 1"),
        required("stride", &C::stride).check(atLeast(1u), "must be >= 1"),
        field("padding", &C::padding)
            .check([](uint32_t p, const C& c) { return p * 2 <= c.kernelSize; }, "must be at most half of kernel_size"),
        field("ceil_mode", &C::ceilMode),
        field("count_include_pad", &C::countIncludePad),
        field("channels_last", &C::channelsLast));
}

}   // namespace


static void bindPool2d(nb::module_& m)
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
PYVAI_NODE_BINDING(bindPool2d);

}   // namespace pyvai
