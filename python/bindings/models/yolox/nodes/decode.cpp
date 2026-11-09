#include "python/bindings/models/yolox/nodes/nodes.h"
#include "models/yolox/nodes/decode.h"
#include <cstdint>
#include <optional>

namespace pyvai::yolox_nodes {

void bindDecode(nb::module_& m)
{
    nb::class_<yolox::DecodeNode, vai::Node>(m, "DecodeNode")
        .def(
            "__init__",
            [](yolox::DecodeNode* self, uint32_t image_size, uint32_t num_classes, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("DecodeNode");
                if (image_size < 32u || image_size % 32u != 0u)
                    throw nb::value_error("DecodeNode: image_size must be a positive multiple of 32");
                if (num_classes == 0u)
                    throw nb::value_error("DecodeNode: num_classes must be >= 1");
                new (self) yolox::DecodeNode(yolox::DecodeNode::Config{
                    .imageSize = image_size,
                    .numClasses = num_classes,
                    .oDtype = o_dtype.value_or(vai::Dtype::None),
                });
            },
            nb::arg("image_size") = 640u, nb::arg("num_classes") = 80u, nb::arg("o_dtype") = nb::none(),
            "YOLOX head decode. Inputs regobj_<s> [5, G, G] and cls_<s> [num_classes, G, G] "
            "for s = p3, p4, p5 (G = image_size / 8, 16, 32); out0 [N, 5 + num_classes] "
            "holds one row per cell, P3 first and cells row-major: cx, cy add the cell's "
            "x, y and w, h take exp, all four times the stride; obj and the class scores "
            "take sigmoid.");
}

}   // namespace pyvai::yolox_nodes
