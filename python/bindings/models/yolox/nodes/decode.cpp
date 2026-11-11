#include "python/bindings/nodes/bind-node.h"
#include "models/yolox/nodes/decode.h"
#include <cstdint>

namespace pyvai {

static void bindDecode(nb::module_& m)
{
    using C = yolox::DecodeNode::Config;
    bindNode<yolox::DecodeNode>(
        m, "DecodeNode",
        "YOLOX head decode. Inputs regobj_<s> [5, G, G] and cls_<s> [num_classes, G, G] "
        "for s = p3, p4, p5 (G = image_size / 8, 16, 32); out0 [N, 5 + num_classes] "
        "holds one row per cell, P3 first and cells row-major: cx, cy add the cell's "
        "x, y and w, h take exp, all four times the stride; obj and the class scores "
        "take sigmoid.",
        field("image_size", &C::imageSize)
            .check([](uint32_t s) { return s >= 32u && s % 32u == 0u; }, "must be a positive multiple of 32"),
        field("num_classes", &C::numClasses).check(atLeast(1u), "must be >= 1"), field("o_dtype", &C::oDtype));
}
PYVAI_MODEL_NODE_BINDING(yolox, bindDecode);

}   // namespace pyvai
