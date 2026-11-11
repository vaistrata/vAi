#include "nodes.h"
#include "vai/nodes/activation.h"
#include <optional>

namespace pyvai {

void bindActivation(nb::module_& m)
{
    // `epilogue` is required (C++ defaults it to None): a standalone no-op activation
    // is almost surely a bug.
    nb::class_<vai::node::Activation, vai::Node>(m, "ActivationNode")
        .def(
            "__init__",
            [](vai::node::Activation* self, vai::Epilogue epilogue, bool in_place, std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("ActivationNode");
                new (self) vai::node::Activation(vai::node::Activation::Config{
                    .epilogue = epilogue,
                    .inPlace = in_place,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("epilogue"), nb::arg("in_place") = false, nb::arg("o_dtype") = nb::none(),
            "Element-wise epilogue over in0. in_place=True aliases out0 onto in0's "
            "buffer, so o_dtype (None: the net's o_default_dtype) must be in0's dtype.");
}

}   // namespace pyvai
