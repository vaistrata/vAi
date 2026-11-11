#include "nodes.h"
#include "vai/nodes/layer-norm.h"
#include <optional>

namespace pyvai {

void bindLayerNorm(nb::module_& m)
{
    nb::class_<vai::node::LayerNorm, vai::Node>(m, "LayerNormNode")
        .def(
            "__init__",
            [](vai::node::LayerNorm* self, float eps, bool coeffs_as_inputs, bool accept_residual,
               std::optional<vai::Dtype> o_dtype) {
                requireCurrentNet("LayerNormNode");
                new (self) vai::node::LayerNorm(vai::node::LayerNorm::Config{
                    .eps = eps,
                    .coeffsAsInputs = coeffs_as_inputs,
                    .acceptResidual = accept_residual,
                });
                self->setODtype(o_dtype.value_or(vai::Dtype::None), dtypeKey());
            },
            nb::arg("eps") = 1e-5f, nb::arg("coeffs_as_inputs") = false, nb::arg("accept_residual") = false,
            nb::arg("o_dtype") = nb::none(),
            "Normalize an [..., feature] input (rank >= 2) over its last axis: "
            "(x - mean) * rsqrt(var + eps) * scale + shift. scale / shift are "
            "parameters, or inputs computed upstream with coeffs_as_inputs=True; "
            "shift applies when filled or wired. accept_residual opens the `residual` input (in0's shape and "
            "dtype), added to the output. Row statistics accumulate in fp32 whatever "
            "the I/O dtypes.");
}

}   // namespace pyvai
