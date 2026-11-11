#include "python/bindings/models/yolox/nodes/nodes.h"
#include "models/experimental/yolox/nodes/sppf.h"
#include <optional>

namespace yolox = model::experimental::yolox;

namespace pyvai::yolox_nodes {

void bindSPPF(nb::module_& m)
{
    nb::class_<yolox::node::SPPF, vai::Node>(m, "SPPFNode")
        .def(
            "__init__",
            [](yolox::node::SPPF* self, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("SPPFNode");
                new (self) yolox::node::SPPF();
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("o_dtype") = nb::none(),
            "YOLOX SPP. in0 [C, H, W] with H * W <= 1024; out0 [4C, H, W] stacks in0, p(in0), "
            "p(p(in0)) and p(p(p(in0))) on channels, p a 5x5 stride-1 max pool that ignores "
            "samples outside the input.");
}

}   // namespace pyvai::yolox_nodes
